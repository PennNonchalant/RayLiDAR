struct Ray
{
    float3 Origin;
    float TMin;

    float3 Direction;
    float TMax;
};

struct Hit
{
    float T;
    uint Valid;

    uint InstanceId;
    uint PrimitiveId;

    float2 Barycentrics;
};

struct TraceParameters
{
    uint RayCount;
};

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure scene;

[[vk::binding(1, 0)]]
StructuredBuffer<Ray> rays;

[[vk::binding(2, 0)]]
RWStructuredBuffer<Hit> hits;

[[vk::push_constant]]
ConstantBuffer<TraceParameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint index = id.x;

    if (index >= parameters.RayCount)
    {
        return;
    }

    Ray ray = rays[index];

    Hit hit;
    hit.T = ray.TMax;
    hit.Valid = 0;
    hit.InstanceId = 0xFFFFFFFFu;
    hit.PrimitiveId = 0xFFFFFFFFu;
    hit.Barycentrics = float2(0.0f, 0.0f);

    hits[index] = hit;
}