// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXOptimizers

// MARK: - KVCache Protocol and Implementations

/// Interface for Key/Value cache for LLMs.
///
/// See ``LanguageModel/newCache(parameters:)``
public protocol KVCache: Evaluatable {

    /// get the current offset
    var offset: Int { get }
    var maxSize: Int? { get }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
}

/// See https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/base.py#L11
public class KVCacheSimple: KVCache, Evaluatable, CustomDebugStringConvertible {
    var keys: MLXArray?
    var values: MLXArray?

    public var maxSize: Int? = nil
    public var offset = 0
    var step = 256

    public init() {}

    public func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += keys.dim(2)

        self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.values?[.ellipsis, previous ..< self.offset, 0...] = values

        return (
            self.keys![.ellipsis, ..<self.offset, 0...],
            self.values![.ellipsis, ..<self.offset, 0...]
        )
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) \(Unmanaged.passUnretained(self).toOpaque()), offset: \(offset), step: \(step), keys: \(keys?.shape.description ?? "-"), values: \(values?.shape.description ?? "-")"
    }
}

/// A placeholder KVCache for quantized models (Added missing type)
public struct QuantizedKVCache: KVCache {
    public var offset: Int
    public var maxSize: Int?
    let groupSize: Int
    let bits: Int

    public init(offset: Int = 0, maxSize: Int? = nil, groupSize: Int, bits: Int) {
        self.offset = offset
        self.maxSize = maxSize
        self.groupSize = groupSize
        self.bits = bits
    }

    /// Placeholder: Quantized cache might not update/store state in the same way.
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("QuantizedKVCache update is not implemented - strategy needed.")
    }

    /// Placeholder: Quantized cache might not have inner state.
    public func innerState() -> [MLXArray] { [] }
}

/// A placeholder KVCache for quantized models (Added missing type)
public struct RotatingKVCache: KVCache {
    public var offset: Int
    public var maxSize: Int?
    let groupSize: Int
    let bits: Int

    public init(offset: Int = 0, maxSize: Int? = nil, groupSize: Int, bits: Int) {
        self.offset = offset
        self.maxSize = maxSize
        self.groupSize = groupSize
        self.bits = bits
    }

    /// Placeholder: Quantized cache might not update/store state in the same way.
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("QuantizedKVCache update is not implemented - strategy needed.")
    }

    /// Placeholder: Quantized cache might not have inner state.
    public func innerState() -> [MLXArray] { [] }
}

func createAdditiveCausalMask(n: Int, offset: Int) -> MLXArray {
    let rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    let linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    let mask = linds[0..., .newAxis] .< rinds[.newAxis]
    return mask * Float32(-1e9)
}

/// create an attention mask using the parameters from the KVCache.
///
/// See also ``MultiHeadAttention/createAdditiveCausalMask(_:dtype:)`` -- same idea
/// but doesn't honor the cache offset.
public func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
    let t = h.dim(1)
    if t > 1 {
        var offset = 0
        if let c = cache?.first {
            offset = c.offset
        }
        return createAdditiveCausalMask(n: t, offset: offset)
            .asType(h.dtype)
    }
    return nil
}

// MARK: - Mask Creation

/// Creates a causal mask for attention mechanisms.
func createCausalMask(
    N: Int,
    offset: Int = 0,
    windowSize: Int? = nil,
    lengths: MLXArray? = nil
) -> MLXArray {
    let rinds = MLXArray(0 ..< (offset + N))
    var linds = (offset > 0) ? MLXArray(offset ..< (offset + N)) : rinds
    linds = linds.expandedDimensions(axes: [-1])
    let rindsExpanded = rinds.expandedDimensions(axes: [0])

    var mask = linds .>= rindsExpanded

    if let windowSize = windowSize {
        mask = mask .&& (linds .<= rindsExpanded + windowSize)
    }

    if var lengths = lengths {
        lengths = lengths.expandedDimensions(axes: [-1, -1, -1])
        mask = mask .&& (rindsExpanded .< lengths)
    }

    return mask
}

/// Creates an attention mask based on input shape and an optional cache.
func createAttentionMask(
    h: MLXArray,
    cache: KVCache? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let T = h.dim(1)

    if T > 1 {
        var offset = 0
        var windowSize: Int? = nil
        var shouldReturnArray = returnArray

        if let cache = cache {
            offset = cache.offset
            // Fixed: Use maxSize from KVCache protocol
            if let maxSize = cache.maxSize {
                windowSize = maxSize
                offset = min(windowSize ?? Int.max, offset)
                shouldReturnArray = shouldReturnArray || (offset + T > (windowSize ?? Int.max))
            }
        }

        if shouldReturnArray {
            return .array(createCausalMask(N: T, offset: offset, windowSize: windowSize))
        } else {
            return .causal
        }
    } else {
        return .none
    }
}

// MARK: - Attention Functions

/// Represents quantized tensors (weights, scales, biases).
typealias QuantizedTensor = (weights: MLXArray, scales: MLXArray, biases: MLXArray)

/// Performs scaled dot-product attention with quantized keys and values.
func quantizedScaledDotProductAttention(
    queries: MLXArray,
    qKeys: QuantizedTensor,
    qValues: QuantizedTensor,
    scale: Float,
    mask: MLXArray?,
    groupSize: Int = 64,
    bits: Int = 8
) throws -> MLXArray {

    let (B, nQHeads, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
    let nKVHeads = qKeys.weights.dim(-3)
    let nRepeats = nQHeads / nKVHeads

    var currentQueries = queries * scale
    var currentQKeys = qKeys
    var currentQValues = qValues

    if nRepeats > 1 {
        currentQueries = currentQueries.reshaped(B, nKVHeads, nRepeats, L, D)
        currentQKeys = (
            qKeys.weights.expandedDimensions(axes: [-3]),
            qKeys.scales.expandedDimensions(axes: [-3]),
            qKeys.biases.expandedDimensions(axes: [-3])
        )
        currentQValues = (
            qValues.weights.expandedDimensions(axes: [-3]),
            qValues.scales.expandedDimensions(axes: [-3]),
            qValues.biases.expandedDimensions(axes: [-3])
        )
    }

    var scores = try MLX.quantizedMatmul(
        currentQueries, currentQKeys.weights,
        scales: currentQKeys.scales,
        biases: currentQKeys.biases,
        transpose: true,
        groupSize: groupSize,
        bits: bits
    )

    if let currentMask = mask {
        if currentMask.dtype == .bool {
            // Fixed: Create fill value as MLXArray and use MLX.where without labels
            //            let fillValue = MLXArray(-Float.greatestFiniteMagnitude, dtype: scores.dtype)
            //            let fillArray = MLXArray.full(scores.shape, values: fillValue, dtype: scores.dtype)
            scores = MLX.where(
                currentMask, scores, scores.dtype.finfo?.min ?? -Double.greatestFiniteMagnitude)
        } else {
            scores = scores + currentMask
        }
    }

    scores = MLX.softmax(scores, axis: -1, precise: true)

    var out = try MLX.quantizedMatmul(
        scores, currentQValues.weights,
        scales: currentQValues.scales,
        biases: currentQValues.biases,
        transpose: false,
        groupSize: groupSize,
        bits: bits
    )

    if nRepeats > 1 {
        out = out.reshaped(B, nQHeads, L, D)
    }

    return out
}

/// Performs scaled dot-product attention, choosing between quantized and standard versions.
func scaledDotProductAttention(
    queries: MLXArray,
    keys: Any,
    values: Any,
    cache: KVCache?,
    scale: Float,
    mask: MLXArray?
) throws -> MLXArray {
    // Check if we should use the quantized version
    if let qCache = cache as? QuantizedKVCache,
        let qKeys = keys as? QuantizedTensor,
        let qValues = values as? QuantizedTensor
    {
        return try quantizedScaledDotProductAttention(
            queries: queries,
            qKeys: qKeys,
            qValues: qValues,
            scale: scale,
            mask: mask,
            groupSize: qCache.groupSize,
            bits: qCache.bits
        )
    }
    // Otherwise, use the standard version (MLXFast)
    else if let keys = keys as? MLXArray, let values = values as? MLXArray {
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
    } else {
        fatalError(
            "Unsupported key/value types or cache combination for scaledDotProductAttention.")
    }
}
