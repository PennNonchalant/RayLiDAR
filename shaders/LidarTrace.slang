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
    RayDesc desc;
    desc.Origin = ray.Origin;
    desc.Direction = ray.Direction;
    desc.TMin = ray.TMin;
    desc.TMax = ray.TMax;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, desc);
    query.Proceed();

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        hit.T = query.CommittedRayT();
        hit.Valid = 1;
        hit.InstanceId = query.CommittedInstanceID();
        hit.PrimitiveId = query.CommittedPrimitiveIndex();
        hit.Barycentrics = query.CommittedTriangleBarycentrics();
    }
    hits[index] = hit;
}