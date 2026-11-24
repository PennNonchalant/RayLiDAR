[vk::constant_id(0)]
const uint kHitBatchSize = 16u;

[vk::constant_id(1)]
const bool kCandidateAlphaBound = true;

[vk::constant_id(2)]
const bool kPrefixAlphaBound = true;

[vk::constant_id(3)]
const uint kWindowCapacity = 1u;

[vk::constant_id(4)]
const float kWindowRadius = 0.0f;

[vk::constant_id(5)]
const float kNormalCosine = 1.0f;

[vk::constant_id(6)]
const float kPlaneTolerance = 0.0f;

static const bool kWeightGeometry = kWindowCapacity > 1u;

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
    uint GaussianIndexStride;
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

[[vk::binding(4, 0)]]
StructuredBuffer<uint> gaussianIndices;

[[vk::push_constant]]
ConstantBuffer<GaussianTraceParameters> parameters;

struct GaussianCandidate
{
    float T;
    uint GaussianIndex;
    float Alpha;
};

struct WindowCandidate
{
    float T;
    float Alpha;
    float3 Normal;
    float NormalProjection;
    float TotalAlpha;
    float CompatibleAlpha;
};

bool Accumulate(float t, float alpha, inout float accumulatedAlpha, inout Hit hit)
{
    accumulatedAlpha += (1.0f - accumulatedAlpha) * alpha;
    if (accumulatedAlpha >= parameters.AlphaThreshold)
    {
        hit.T = t;
        hit.Valid = 1;
        return true;
    }
    return false;
}

bool ConsumeWindow(WindowCandidate candidate, inout uint head, inout uint count, inout float accumulatedAlpha,
                   inout Hit hit)
{
    head = (head + 1u) & (kWindowCapacity - 1u);
    --count;
    float alpha = candidate.Alpha == 0.0f ? 0.0f : candidate.Alpha * (candidate.CompatibleAlpha / candidate.TotalAlpha);
    return Accumulate(candidate.T, alpha, accumulatedAlpha, hit);
}

bool IsBefore(float leftT, uint leftIndex, float rightT, uint rightIndex)
{
    return leftT < rightT || (leftT == rightT && leftIndex < rightIndex);
}

float ResumeTMin(float currentMin, float hardwareT)
{
    return hardwareT > currentMin ? max(currentMin, asfloat(asuint(hardwareT) - 1u)) : currentMin;
}

float CumulativeAlphaBound(GaussianCandidate candidates[kHitBatchSize], uint candidateCount, float accumulatedAlpha,
                           float tMax)
{
    for (uint slot = 0; slot < candidateCount; ++slot)
    {
        GaussianCandidate candidate = candidates[slot];
        if (candidate.T >= tMax)
        {
            break;
        }
        accumulatedAlpha += (1.0f - accumulatedAlpha) * candidate.Alpha;
        if (accumulatedAlpha >= parameters.AlphaThreshold)
        {
            return candidate.T;
        }
    }
    return tMax;
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
    WindowCandidate window[kWindowCapacity];
    uint windowHead = 0;
    uint windowCount = 0;

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
            uint gaussianIndex = primitive / 2u;
            if (parameters.GaussianIndexStride != 0u)
            {
                uint batchSlot = query.CandidateRayInstanceCustomIndex() * parameters.GaussianIndexStride + gaussianIndex;
                gaussianIndex = gaussianIndices[batchSlot];
            }
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

            uint previousCandidateCount = candidateCount;
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

            if (!kWeightGeometry && kCandidateAlphaBound &&
                accumulatedAlpha + (1.0f - accumulatedAlpha) * alpha >= parameters.AlphaThreshold)
            {
                batchTMax = min(batchTMax, t);
            }
            if (candidateCount == kHitBatchSize)
            {
                batchTMax = min(batchTMax, candidates[kHitBatchSize - 1].T);
            }
            if (!kWeightGeometry && kPrefixAlphaBound && candidateCount > 1u && candidateCount > previousCandidateCount &&
                ((candidateCount & (candidateCount - 1u)) == 0u || candidateCount == kHitBatchSize))
            {
                batchTMax = CumulativeAlphaBound(candidates, candidateCount, accumulatedAlpha, batchTMax);
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
            if (kWeightGeometry)
            {
                while (windowCount != 0u && candidate.T - window[windowHead].T > kWindowRadius)
                {
                    if (ConsumeWindow(window[windowHead], windowHead, windowCount, accumulatedAlpha, hit))
                    {
                        hits[index] = hit;
                        return;
                    }
                }

                Gaussian gaussian = gaussians[candidate.GaussianIndex];
                float3 normal = normalize(cross(gaussian.AxisU, gaussian.AxisV));
                float normalProjection = abs(dot(normal, ray.Direction));
                float totalAlpha = 0.0f;
                float compatibleAlpha = 0.0f;
                for (uint offset = 0; offset < windowCount; ++offset)
                {
                    uint neighborSlot = (windowHead + offset) & (kWindowCapacity - 1u);
                    WindowCandidate neighbor = window[neighborSlot];
                    totalAlpha += neighbor.Alpha;
                    window[neighborSlot].TotalAlpha += candidate.Alpha;
                    float planeDistance = (candidate.T - neighbor.T) * max(normalProjection, neighbor.NormalProjection);
                    if (planeDistance <= kPlaneTolerance && abs(dot(normal, neighbor.Normal)) >= kNormalCosine)
                    {
                        compatibleAlpha += neighbor.Alpha;
                        window[neighborSlot].CompatibleAlpha += candidate.Alpha;
                    }
                }

                uint tail = (windowHead + windowCount) & (kWindowCapacity - 1u);
                window[tail] = {candidate.T, candidate.Alpha, normal, normalProjection, totalAlpha + candidate.Alpha,
                                compatibleAlpha + candidate.Alpha};
                ++windowCount;
                if (windowCount == kWindowCapacity &&
                    ConsumeWindow(window[windowHead], windowHead, windowCount, accumulatedAlpha, hit))
                {
                    hits[index] = hit;
                    return;
                }
            }
            else if (Accumulate(candidate.T, candidate.Alpha, accumulatedAlpha, hit))
            {
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

    while (kWeightGeometry && windowCount != 0u)
    {
        if (ConsumeWindow(window[windowHead], windowHead, windowCount, accumulatedAlpha, hit))
        {
            break;
        }
    }
    hits[index] = hit;
}
