//
//  Gemma3TextConfiguration.swift
//  mlx-swift-examples
//
//  Created by Yangming(Ocean) Zhang on 5/29/25.
//


import Foundation
import MLX
import MLXLMCommon
import MLXNN


// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma3_text.py

// MARK: - Configuration

public struct Gemma3TextConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeGlobalBaseFreq: Float
    var ropeLocalBaseFreq: Float
    var ropeTraditional: Bool
    var queryPreAttnScalar: Float
    var slidingWindow: Int
    var slidingWindowPattern: Int
    var finalLogitSoftcapping: Float?


    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeGlobalBaseFreq = "rope_theta"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case finalLogitSoftcapping = "final_logit_softcapping"
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)

        // Default values with optional decoding
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262208
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        ropeGlobalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeGlobalBaseFreq) ?? 1_000_000.0
        ropeLocalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar =
            try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256  // 0.0625 ?
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 4096
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
        finalLogitSoftcapping = try container.decodeIfPresent(
            Float.self, forKey: .finalLogitSoftcapping)
    }
}

// MARK: - Layers

private class RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1.0 + weight, eps: eps)
    }
}

private class Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let headDim: Int
    let scale: Float
    let layerIdx: Int
    let isSliding: Bool
    
    @ModuleInfo(key: "q_proj") var q_proj: Linear
    @ModuleInfo(key: "k_proj") var k_proj: Linear
    @ModuleInfo(key: "v_proj") var v_proj: Linear
    @ModuleInfo(key: "o_proj") var o_proj: Linear
    
    @ModuleInfo(key: "q_norm") var q_norm: RMSNorm
    @ModuleInfo(key: "k_norm") var k_norm: RMSNorm
    
    let rope: RoPE

    init(_ args: Gemma3TextConfiguration, layerIdx: Int) {
        let dim = args.hiddenSize
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.repeats = nHeads / nKVHeads
        self.headDim = args.headDim
        self.layerIdx = layerIdx
        
        self.scale = pow(args.queryPreAttnScalar, -0.5)
        
        self._q_proj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._k_proj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._v_proj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._o_proj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
        
        self._q_norm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self._k_norm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self.isSliding = (layerIdx + 1) % args.slidingWindowPattern != 0

        self.rope = RoPE(
            dimensions: headDim,
            traditional: args.ropeTraditional,
            base: isSliding ? args.ropeLocalBaseFreq : args.ropeGlobalBaseFreq
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let (B, L, _) = x.shape3

        var queries = q_proj(x)
        var keys = k_proj(x)
        var values = v_proj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)
        
        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }
        
        // Sliding window
        var currentMask = mask.masks?.first
        if let m = mask.masks?.first, m.dim(-1) != keys.dim(-2) {
            currentMask = m[.ellipsis, (-keys.dim(-2))...]
        }

        var output = try scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: currentMask
        )
        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return o_proj(output)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo var gate_proj: Linear
    @ModuleInfo var down_proj: Linear
    @ModuleInfo var up_proj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gate_proj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down_proj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up_proj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down_proj(MLXNN.gelu(gate_proj(x)) * up_proj(x))
    }
}

// Equivalent of Python's `clip_residual`
private func clipResidual(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
    if x.dtype != .float16 {
        return x + y
    }
    let bound = MLX.DType.float16.finfo!.max
    return clip(x.asType(.float32) + y.asType(.float32), min: -bound, max: bound).asType(.float16)
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var self_attn: Attention
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var input_layernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var post_attention_layernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var pre_feedforward_layernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var post_feedforward_layernorm: RMSNorm

    init(_ args: Gemma3TextConfiguration, layerIdx: Int) {
        self._self_attn.wrappedValue = Attention(args, layerIdx: layerIdx)
        self._mlp.wrappedValue = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._input_layernorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._post_attention_layernorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._pre_feedforward_layernorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._post_feedforward_layernorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        var r = self_attn(input_layernorm(x), mask: mask, cache: cache)
        let h = clipResidual(x, post_attention_layernorm(r))
        r = mlp(pre_feedforward_layernorm(h))
        let out = clipResidual(h, post_feedforward_layernorm(r))
        return out
    }
}

private class Gemma3TextModelInner: Module {
    let config: Gemma3TextConfiguration
    var layers: [TransformerBlock]
    
    @ModuleInfo var embed_tokens: Embedding
    @ModuleInfo var norm: RMSNorm

    init(_ args: Gemma3TextConfiguration) {
        self.config = args
        self.layers = (0..<args.hiddenLayers).map { TransformerBlock(args, layerIdx: $0) }
        self._embed_tokens.wrappedValue = MLXNN.Embedding(embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?, mask: MLXFast.ScaledDotProductAttentionMaskMode, inputEmbeddings: MLXArray? = nil) -> MLXArray {
        var h: MLXArray
        if let inputEmbeddings {
            h = inputEmbeddings
        } else {
            h = embed_tokens(inputs)
        }
        h = h * MLXArray(pow(Float(config.hiddenSize), 0.5)).asType(h.dtype)
            
        var currentMask = mask
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        var slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        
        let pattern = config.slidingWindowPattern - 1
        print("mask?: \(mask ?? nil)")
        if case .none = mask {
            let globalCache: [KVCache]? = cache.flatMap { c in
                let globalLayerIndex = config.slidingWindowPattern - 1
                return globalLayerIndex < c.count ? [c[globalLayerIndex]] : nil
            }
            fullMask = createAttentionMaskFast(h: h, cache: globalCache)
            slidingWindowMask = createAttentionMaskFast(h: h, cache: cache)
        }


        for (i, layer) in layers.enumerated() {
            let isGlobal = (i + 1) % config.slidingWindowPattern == 0
            if mask == nil {
                 currentMask = isGlobal ? fullMask : slidingWindowMask
            }
            h = layer(h, mask: slidingWindowMask, cache: cache![i])
        }
        return norm(h)
    }
}

// MARK: - Model

public class Gemma3TextModel: Module, LLMModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    
    let modelType: String
    let configuration: Gemma3TextConfiguration
    private let model: Gemma3TextModelInner
    
    @ModuleInfo var lm_head: Linear

    public init(_ args: Gemma3TextConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        
        self.configuration = args
        self.modelType = args.modelType
        self.model = Gemma3TextModelInner(args)
        
        self._lm_head.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(inputs, cache: cache, mask: .none, inputEmbeddings: nil)
    }
    
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil, mask: MLXFast.ScaledDotProductAttentionMaskMode, inputEmbeddings: MLXArray? = nil) -> MLXArray {
        let out = model(inputs, cache: cache, mask: mask, inputEmbeddings: inputEmbeddings)
        return lm_head(out)
    }
    
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights
        if newWeights["lm_head.weight"] == nil {
            newWeights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        }
        return newWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        var caches = [KVCache]()
        for i in 0..<configuration.hiddenLayers {
            if (i + 1) % configuration.slidingWindowPattern == 0 {
                caches.append(KVCacheBase())
            } else {
                // Assuming RotatingKVCache is defined elsewhere and takes maxSize & keep
                // For now, using KVCacheSimple as a placeholder if RotatingKVCache is not available
                // or if the parameters are not fully defined for it here.
                // If RotatingKVCache is intended, ensure it's correctly implemented and initialized.
//                 caches.append(RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)) // Example params
                caches.append(KVCacheBase())
            }
        }
        return caches
    }
}


extension Gemma3TextModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.self_attn, ["q_proj", "v_proj"]) }
    }
}
