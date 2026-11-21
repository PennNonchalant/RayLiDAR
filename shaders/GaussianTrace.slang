[vk::constant_id(0)]
const uint kHitBatchSize = 1u;

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

struct Gaussian
{
    float3 Center;
    float Opacity;
    float3 AxisU;
    float SigmaU;
    float3 AxisV;
    float SigmaV;
};

struct GaussianTraceParameters
{
    uint RayCount;
    uint GaussiansPerBatch;
    float SigmaCutoff;
    float AlphaThreshold;
};

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure scene;

[[vk::binding(1, 0)]]
StructuredBuffer<Ray> rays;

[[vk::binding(2, 0)]]
RWStructuredBuffer<Hit> hits;

[[vk::binding(3, 0)]]
StructuredBuffer<Gaussian> gaussians;

[[vk::push_constant]]
ConstantBuffer<GaussianTraceParameters> parameters;

struct GaussianCandidate
{
    float T;
    uint GaussianIndex;
    float Alpha;
};

bool IsBefore(float leftT, uint leftIndex, float rightT, uint rightIndex)
{
    return leftT < rightT || (leftT == rightT && leftIndex < rightIndex);
}

float ResumeTMin(float currentMin, float hardwareT)
{
    return hardwareT > currentMin ? max(currentMin, asfloat(asuint(hardwareT) - 1u)) : currentMin;
}

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
    RayDesc desc;
    desc.Origin = ray.Origin;
    desc.Direction = ray.Direction;
    desc.TMin = ray.TMin;
    desc.TMax = ray.TMax;

    Hit hit;
    hit.T = ray.TMax;
    hit.Valid = 0;
    hit.InstanceId = 0xFFFFFFFFu;
    hit.PrimitiveId = 0xFFFFFFFFu;
    hit.Barycentrics = float2(0.0f, 0.0f);

    float cutoffSquared = parameters.SigmaCutoff * parameters.SigmaCutoff;
    float accumulatedAlpha = 0.0f;
    float lastT = -1.0f;
    uint lastGaussianIndex = 0;

    while (true)
    {
        GaussianCandidate candidates[kHitBatchSize];
        uint candidateCount = 0;
        bool hasMore = false;
        float batchTMax = desc.TMax;

        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, desc);
        while (query.Proceed())
        {
            float t = query.CandidateTriangleRayT();
            if (t > batchTMax)
            {
                query.CommitNonOpaqueTriangleHit();
                hasMore = true;
                continue;
            }

            uint primitive = query.CandidateRayPrimitiveIndex();
            uint gaussianIndex = query.CandidateRayInstanceCustomIndex() * parameters.GaussiansPerBatch + primitive / 2u;
            if (!IsBefore(lastT, lastGaussianIndex, t, gaussianIndex))
            {
                continue;
            }

            float2 b = query.CandidateTriangleBarycentrics();
            float2 uv = (primitive & 1u) == 0u ? float2(b.x + b.y, b.y) : float2(b.x, b.x + b.y);
            uv = 2.0f * uv - 1.0f;
            float radiusSquared = cutoffSquared * dot(uv, uv);
            if (radiusSquared > cutoffSquared)
            {
                continue;
            }
            float alpha = gaussians[gaussianIndex].Opacity * exp(-0.5f * radiusSquared);

            uint slot = 0;
            while (slot < candidateCount && candidates[slot].GaussianIndex != gaussianIndex)
            {
                ++slot;
            }
            if (slot < candidateCount)
            {
                if (t > candidates[slot].T)
                {
                    continue;
                }
                if (t == candidates[slot].T)
                {
                    alpha = max(alpha, candidates[slot].Alpha);
                }
            }
            else if (candidateCount == kHitBatchSize)
            {
                hasMore = true;
                slot = candidateCount - 1;
                if (!IsBefore(t, gaussianIndex, candidates[slot].T, candidates[slot].GaussianIndex))
                {
                    continue;
                }
            }
            else
            {
                ++candidateCount;
            }

            while (slot > 0 && IsBefore(t, gaussianIndex, candidates[slot - 1].T, candidates[slot - 1].GaussianIndex))
            {
                candidates[slot] = candidates[slot - 1];
                --slot;
            }
            candidates[slot] = {t, gaussianIndex, alpha};

            if (accumulatedAlpha + (1.0f - accumulatedAlpha) * alpha >= parameters.AlphaThreshold)
            {
                batchTMax = min(batchTMax, t);
            }
            if (candidateCount == kHitBatchSize)
            {
                batchTMax = min(batchTMax, candidates[kHitBatchSize - 1].T);
            }
        }

        for (uint slot = 0; slot < candidateCount; ++slot)
        {
            GaussianCandidate candidate = candidates[slot];
            if (candidate.T > batchTMax)
            {
                hasMore = true;
                break;
            }
            lastT = candidate.T;
            lastGaussianIndex = candidate.GaussianIndex;
            accumulatedAlpha += (1.0f - accumulatedAlpha) * candidate.Alpha;
            if (accumulatedAlpha >= parameters.AlphaThreshold)
            {
                hit.T = candidate.T;
                hit.Valid = 1;
                hits[index] = hit;
                return;
            }
        }

        if (!hasMore)
        {
            break;
        }
        desc.TMin = ResumeTMin(desc.TMin, lastT);
    }

    hits[index] = hit;
}
