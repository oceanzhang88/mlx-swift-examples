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
    func isTrimmable() -> Bool
    func trim(numTokens: Int) -> Int
}

public extension KVCache {
    // Default implementation for non-trimmable caches
    func isTrimmable() -> Bool {
        return false
    }
    
    func trim(numTokens: Int) -> Int {
        return 0
    }
}

/// See https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/base.py#L11

// MARK: - KVCache

public class KVCacheBase: KVCache {
    public var maxSize: Int?
    
    public var keys: MLXArray?
    public var values: MLXArray?
    public var offset: Int = 0
    var step: Int = 256
    
    public init() {}
    
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset
        if self.keys == nil || (prev + keys.dim(2)) > self.keys!.dim(2) {
            let (b, nKvHeads, _, kHeadDim) = keys.shape4
            let vHeadDim = values.dim(3)
            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [b, nKvHeads, nSteps * step, kHeadDim]
            let vShape = [b, nKvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)
            
            if var currentKeys = self.keys, var currentValues = self.values {
                if prev % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prev, 0...]
                    currentValues = currentValues[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }
        
        offset += keys.dim(2)
        self.keys?[.ellipsis, prev ..< offset, 0...] = keys
        self.values?[.ellipsis, prev ..< offset, 0...] = values
        
        return (self.keys![.ellipsis, ..<offset, 0...], self.values![.ellipsis, ..<offset, 0...])
    }
    
    public func innerState() -> [MLX.MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }
    
    public func isTrimmable() -> Bool {
        return true
    }
    
    public func trim(numTokens n: Int) -> Int {
        let trimmedCount = min(offset, n)
        offset -= trimmedCount
        return trimmedCount
    }
    
    public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
        let quantCache = QuantizedKVCache(groupSize: groupSize, bits: bits)
        quantCache.offset = offset
        if let k = keys, let v = values {
            quantCache.keys = MLX.quantized(k, groupSize: groupSize, bits: bits)
            quantCache.values = MLX.quantized(v, groupSize: groupSize, bits: bits)
        }
        return quantCache
    }
}


// MARK: - QuantizedKVCache

public class QuantizedKVCache: KVCache {
    public var maxSize: Int?
    
    public var keys: (MLXArray, MLXArray, MLXArray)?
    public var values: (MLXArray, MLXArray, MLXArray)?
    public var offset: Int = 0
    var step: Int = 256
    let groupSize: Int
    let bits: Int
    
    public init(groupSize: Int = 64, bits: Int = 8) {
        self.groupSize = groupSize
        self.bits = bits
    }
    
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let (b, nKvHeads, numSteps, kHeadDim) = keys.shape4
        let vHeadDim = values.dim(-1)
        let prev = offset
        
        //        if self.keys == nil || (prev + numSteps) > self.keys!.0.dim(-2) {
        //            let elPerInt = 8 * MLXArray.UInt32Scalar.scalarSize / bits
        //            let newSteps = (step + numSteps - 1) / step * step
        //            let shape = [b, nKvHeads, newSteps]
        //
        //            func initQuant(dim: Int) -> (MLXArray, MLXArray, MLXArray) {
        //                return (
        //                    MLXArray.zeros(shape + [dim / elPerInt], type: .uint32),
        //                    MLXArray.zeros(shape + [dim / groupSize], type: keys.dtype),
        //                    MLXArray.zeros(shape + [dim / groupSize], type: keys.dtype)
        //                )
        //            }
        //
        //            func expandQuant(current: (MLXArray, MLXArray, MLXArray)?) -> (MLXArray, MLXArray, MLXArray) {
        //                guard let current = current else {
        //                    // This case should ideally be handled by the outer `if self.keys == nil`
        //                    // For safety, re-initialize if current is nil, though this might indicate a logic error.
        //                    let dim = currentKeys == nil ? kHeadDim : vHeadDim // Simplified logic, needs context
        //                    return initQuant(dim: dim)
        //                }
        //                let newShapeSuffix = current.0.shape.suffix(from: current.0.ndim - 1) // Gets the last dimension
        //                let newX0 = MLXArray.zeros([shape[0], shape[1], shape[2]] + newShapeSuffix, type: current.0.dtype)
        //                let newX1 = MLXArray.zeros([shape[0], shape[1], shape[2]] + newShapeSuffix, type: current.1.dtype)
        //                let newX2 = MLXArray.zeros([shape[0], shape[1], shape[2]] + newShapeSuffix, type: current.2.dtype)
        //
        //
        //                return (
        //                    concatenated([current.0, newX0], axis: -2),
        //                    concatenated([current.1, newX1], axis: -2),
        //                    concatenated([current.2, newX2], axis: -2)
        //                )
        //            }
        //
        //
        //            if var currentKeys = self.keys, var currentValues = self.values {
        //                 if prev % step != 0 {
        //                    currentKeys = (
        //                        currentKeys.0[.ellipsis, ..<prev, 0...],
        //                        currentKeys.1[.ellipsis, ..<prev, 0...],
        //                        currentKeys.2[.ellipsis, ..<prev, 0...]
        //                    )
        //                    currentValues = (
        //                        currentValues.0[.ellipsis, ..<prev, 0...],
        //                        currentValues.1[.ellipsis, ..<prev, 0...],
        //                        currentValues.2[.ellipsis, ..<prev, 0...]
        //                    )
        //                }
        //                self.keys = expandQuant(current: currentKeys)
        //                self.values = expandQuant(current: currentValues)
        //            } else {
        //                self.keys = initQuant(dim: kHeadDim)
        //                self.values = initQuant(dim: vHeadDim)
        //            }
        //        }
        //
        //        offset += numSteps
        //
        //        let quantizedKeys = MLX.quantized(keys, groupSize: groupSize, bits: bits)
        //        let quantizedValues = MLX.quantized(values, groupSize: groupSize, bits: bits)
        //
        //        self.keys!.0[.ellipsis, prev ..< offset, 0...] = quantizedKeys.0
        //        self.keys!.1[.ellipsis, prev ..< offset, 0...] = quantizedKeys.1
        //        self.keys!.2[.ellipsis, prev ..< offset, 0...] = quantizedKeys.2
        //
        //        self.values!.0[.ellipsis, prev ..< offset, 0...] = quantizedValues.0
        //        self.values!.1[.ellipsis, prev ..< offset, 0...] = quantizedValues.1
        //        self.values!.2[.ellipsis, prev ..< offset, 0...] = quantizedValues.2
        //
        //        return (
        //            (self.keys!.0[.ellipsis, ..<offset, 0...],
        //             self.keys!.1[.ellipsis, ..<offset, 0...],
        //             self.keys!.2[.ellipsis, ..<offset, 0...]),
        //            (self.values!.0[.ellipsis, ..<offset, 0...],
        //             self.values!.1[.ellipsis, ..<offset, 0...],
        //             self.values!.2[.ellipsis, ..<offset, 0...])
        //        )
        return (MLXArray.mlxNone, MLXArray.mlxNone)
    }
    
    public func innerState() -> [MLX.MLXArray] {
        //        [self.keys, self.values].compactMap { $0 }
        []
    }
    
    public func isTrimmable() -> Bool {
        return true
    }
    
    public func trim(numTokens n: Int) -> Int {
        let trimmedCount = min(offset, n)
        offset -= trimmedCount
        return trimmedCount
    }
}

// MARK: - RotatingKVCache

public class RotatingKVCache: KVCache {
    public var keys: MLXArray?
    public var values: MLXArray?
    public var offset: Int = 0
    var keep: Int
    public var maxSize: Int?
    var step: Int = 256
    private var idx: Int = 0
    
    public init(maxSize: Int?, keep: Int = 0, step: Int = 256) {
        self.maxSize = maxSize
        self.keep = keep
        self.step = step
    }
    
    private func trimArray(_ arr: MLXArray?, trimSize: Int, append: MLXArray? = nil) -> MLXArray? {
        guard var currentArr = arr else { return append }
        var toCat = [MLXArray]()
        if trimSize > 0 {
            toCat.append(currentArr[.ellipsis, ..<keep, 0...])
            toCat.append(currentArr[.ellipsis, (trimSize + keep)..., 0...])
        } else {
            toCat.append(currentArr)
        }
        if let appendArr = append {
            toCat.append(appendArr)
        }
        return concatenated(toCat, axis: 2)
    }
    
    private func temporalOrder(_ arr: MLXArray?) -> MLXArray? {
        guard var currentArr = arr else { return nil }
        if idx == currentArr.dim(2) {
            return currentArr
        } else if idx < offset {
            return concatenated(
                [
                    currentArr[.ellipsis, ..<keep, 0...],
                    currentArr[.ellipsis, idx..., 0...],
                    currentArr[.ellipsis, keep..<idx, 0...],
                ],
                axis: 2
            )
        } else {
            return currentArr[.ellipsis, ..<idx, 0...]
        }
    }
    
    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            self.keys = temporalOrder(self.keys)
            self.values = temporalOrder(self.values)
            
            let trimSize = idx - (maxSize ?? idx)
            self.keys = trimArray(self.keys, trimSize: trimSize, append: keys)
            self.values = trimArray(self.values, trimSize: trimSize, append: values)
        }
        offset += keys.dim(2)
        idx = self.keys!.dim(2)
        return (self.keys!, self.values!)
    }
    
    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let (b, nKvHeads, s, kHeadDim) = keys.shape4
        let prev = offset
        
        if self.keys == nil || (prev >= self.keys!.dim(2) && self.keys!.dim(2) < (maxSize ?? Int.max)) {
            let vHeadDim = values.dim(3)
            let newSize = min(step, (maxSize ?? Int.max) - prev)
            let kShape = [b, nKvHeads, newSize, kHeadDim]
            let vShape = [b, nKvHeads, newSize, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)
            
            if var currentKeys = self.keys, var currentValues = self.values {
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
            idx = prev
        }
        
        if let currentMaxSize = maxSize {
            let trimSize = self.keys!.dim(2) - currentMaxSize
            if trimSize > 0 {
                self.keys = trimArray(self.keys, trimSize: trimSize)
                self.values = trimArray(self.values, trimSize: trimSize)
                idx = currentMaxSize
            }
        }
        
        if let currentMaxSize = maxSize, idx == currentMaxSize {
            idx = keep
        }
        
        self.keys?[.ellipsis, idx ..< (idx + s), 0...] = keys
        self.values?[.ellipsis, idx ..< (idx + s), 0...] = values
        offset += s
        idx += s
        
        if offset < (maxSize ?? Int.max) {
            return (self.keys![.ellipsis, ..<offset, 0...], self.values![.ellipsis, ..<offset, 0...])
        }
        return (self.keys!, self.values!)
    }
    
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if keys.dim(2) == 1 {
            return updateInPlace(keys: keys, values: values)
        }
        return updateConcat(keys: keys, values: values)
    }
    
    public func innerState() -> [MLX.MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }
    
    public func isTrimmable() -> Bool {
        return maxSize == nil || offset < maxSize!
    }
    
    public func trim(numTokens n: Int) -> Int {
        let trimmedCount = min(offset, n)
        offset -= trimmedCount
        idx -= trimmedCount
        return trimmedCount
    }
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

/// create an attention mask using the parameters from the KVCache.
///
/// See also ``MultiHeadAttention/createAdditiveCausalMask(_:dtype:)`` -- same idea
/// but doesn't honor the cache offset.
@_disfavoredOverload
public func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
    let t = h.dim(1)
    if t > 1 {
        var offset = 0
        if let c = cache?.first {
            offset = c.offset
        }
        return createCausalMask(N: t, offset: offset)
            .asType(h.dtype)
    }
    return nil
}

/// Creates an attention mask based on input shape and an optional cache.
public func createAttentionMaskFast(
    h: MLXArray,
    cache: [KVCache]? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let T = h.dim(1)
    print("createAttentionMaskFast, returnArray: \(returnArray)")
    if T > 1 {
        var offset = 0
        var windowSize: Int = 0
        var shouldReturnArray = returnArray
        
        if let cache = cache?.first {
            offset = cache.offset
            // Fixed: Use maxSize from KVCache protocol
            if let maxSize = cache.maxSize {
                windowSize = maxSize
                offset = min(windowSize, offset)
                shouldReturnArray = shouldReturnArray || (offset + T > windowSize)
            }
        }
        
        print("shouldReturnArray: \(shouldReturnArray)")
        if shouldReturnArray {
            return .array(createCausalMask(N: T, offset: offset, windowSize: windowSize))
        } else {
            return .array(createCausalMask(N: T, offset: offset, windowSize: windowSize))
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
) -> MLXArray {
    
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
public func scaledDotProductAttention(
    queries: MLXArray,
    keys: Any,
    values: Any,
    cache: KVCache?,
    scale: Float,
    mask: MLXArray?
) -> MLXArray {
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

