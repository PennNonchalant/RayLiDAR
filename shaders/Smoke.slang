[[vk::binding(0, 0)]]
StructuredBuffer<uint> inputData;

[[vk::binding(0, 1)]]
RWStructuredBuffer<uint> outputData;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint index = id.x;
    outputData[index] = inputData[index] * 2u + 1u;
}
