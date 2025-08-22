from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'common.onnx'
model_quant = 'common_q8.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
