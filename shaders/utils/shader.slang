interface Shader<VertexInput, InterpolatedVertex> {
    InterpolatedVertex interpolateUsingBarycentrics(VertexInput[3] inputs, float3 bary);
}