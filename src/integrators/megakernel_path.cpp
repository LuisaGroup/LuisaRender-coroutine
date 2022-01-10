//
// Created by Mike Smith on 2022/1/10.
//

#include <luisa-compute.h>
#include <scene/pipeline.h>
#include <scene/integrator.h>

namespace luisa::render {

class MegakernelPathTracing final : public Integrator {

private:
    uint _max_depth;

public:
    MegakernelPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 5u), 1u)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] string_view impl_type() const noexcept override { return "megapath"; }
    [[nodiscard]] unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelPathTracingInstance final : public Integrator::Instance {

private:
    uint _max_depth;

private:
    static void _render_one_camera(
        Stream &stream, Pipeline &pipeline,
        const Camera::Instance *camera,
        const Filter::Instance *filter,
        Film::Instance *film, uint max_depth) noexcept;

public:
    explicit MegakernelPathTracingInstance(const MegakernelPathTracing *node) noexcept
        : Integrator::Instance{node}, _max_depth{node->max_depth()} {}
    void render(Stream &stream, Pipeline &pipeline) noexcept override {
        for (auto i = 0u; i < pipeline.camera_count(); i++) {
            auto [camera, film, filter] = pipeline.camera(i);
            _render_one_camera(stream, pipeline, camera, filter, film, _max_depth);
            film->save(stream, camera->node()->file());
        }
    }
};

unique_ptr<Integrator::Instance> MegakernelPathTracing::build(Pipeline &, CommandBuffer &) const noexcept {
    return luisa::make_unique<MegakernelPathTracingInstance>(this);
}

void MegakernelPathTracingInstance::_render_one_camera(
    Stream &stream, Pipeline &pipeline, const Camera::Instance *camera,
    const Filter::Instance *filter, Film::Instance *film, uint max_depth) noexcept {
    auto spp = camera->node()->spp();
    auto resolution = film->node()->resolution();
    auto image_file = camera->node()->file();
    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    auto sampler = pipeline.sampler();
    auto command_buffer = stream.command_buffer();
    film->clear(command_buffer);
    sampler->reset(command_buffer, resolution, spp);
    command_buffer.commit();

    using namespace luisa::compute;
    Kernel2D render_kernel = [&](UInt frame_index, Float4x4 camera_to_world, Float3x3 camera_to_world_normal, Float time) noexcept {

        set_block_size(8u, 8u, 1u);

        auto pixel_id = dispatch_id().xy();
        sampler->start(pixel_id, frame_index);
        auto pixel = make_float2(pixel_id);
        auto throughput = def(make_float3(1.0f));
        auto [filter_offset, filter_weight] = filter->sample(*sampler);
        pixel += filter_offset;
        throughput *= filter_weight;

        auto [camera_ray, camera_weight] = camera->generate_ray(*sampler, pixel, time);
        camera_ray.origin = make_float3(camera_to_world * make_float4(def<float3>(camera_ray.origin), 1.0f));
        camera_ray.direction = normalize(camera_to_world_normal * def<float3>(camera_ray.direction));
        throughput *= camera_weight;

        auto ray = camera_ray;
        auto radiance = def(make_float3(0.0f));
        auto interaction = pipeline.intersect(ray);
        $if(interaction->valid()) {
            auto has_light = (interaction->shape()->light_flags() & Light::property_flag_black) == 0u;
            auto has_material = (interaction->shape()->material_flags() & Material::property_flag_black) == 0u;
            $if(has_material) {

                // sample light
                constexpr auto light_position = make_float3(-0.24f, 1.98f, 0.16f);
                constexpr auto light_u = make_float3(-0.24f, 1.98f, -0.22f) - light_position;
                constexpr auto light_v = make_float3(0.23f, 1.98f, 0.16f) - light_position;
                constexpr auto light_emission = make_float3(17.0f, 12.0f, 4.0f);
                auto light_area = length(cross(light_u, light_v));
                auto light_normal = normalize(cross(light_u, light_v));
                auto u_light = sampler->generate_2d();
                auto p_light = light_position + u_light.x * light_u + u_light.y * light_v;
                auto shadow_ray = interaction->spawn_ray_to(p_light);
                auto wi = def<float3>(shadow_ray.direction);
                auto light_front_face = dot(wi, light_normal) < 0.0f;
                auto occluded = pipeline.intersect_any(shadow_ray);
                $if(light_front_face & !occluded) {
                    pipeline.decode_material(*interaction, [&](const Material::Closure &material) {
                        auto [f, pdf] = material.evaluate(wi);
                        radiance = throughput * ite(pdf > 1e-4f, f, 0.0f) *
                                   abs(dot(interaction->shading().n(), wi)) *
                                   light_emission * light_area;
                    });
                };
            };
        };
        film->accumulate(pixel_id, radiance);
        sampler->save_state();
    };

    auto render = pipeline.device().compile(render_kernel);
    stream << synchronize();
    Clock clock;
    auto time_start = camera->node()->time_span().x;
    auto time_end = camera->node()->time_span().x;
    auto spp_per_commit = 16u;
    for (auto i = 0u; i < spp; i++) {
        auto t = static_cast<float>((static_cast<double>(i) + 0.5f) / static_cast<double>(spp));
        auto time = lerp(time_start, time_end, t);
        pipeline.update_geometry(command_buffer, time);
        auto camera_to_world = camera->node()->transform()->matrix(t);
        auto camera_to_world_normal = transpose(inverse(make_float3x3(camera_to_world)));
        command_buffer << render(i, camera_to_world, camera_to_world_normal, time).dispatch(resolution);
        if (spp % spp_per_commit == spp_per_commit - 1u) [[unlikely]] { command_buffer << commit(); }
    }
    command_buffer << commit();
    stream << synchronize();
    LUISA_INFO("Rendering finished in {} ms.", clock.toc());
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPathTracing)