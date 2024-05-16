//
// Created by Hercier on 2023/12/25.
//

#include <luisa-compute.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <util/progress_bar.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::render {

using namespace compute;

class CoroutinePathTracing final : public ProgressiveIntegrator {

public:
    enum struct Scheduler {
        Simple,
        Wavefront,
        Persistent,
    };
    using SchedulerBase = coroutine::CoroScheduler<float, float, uint>;

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _samples_per_pass;
    Scheduler _scheduler;
    coroutine::WavefrontCoroSchedulerConfig _wavefront_config;
    coroutine::PersistentThreadsCoroSchedulerConfig _persistent_config;

public:
    CoroutinePathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _samples_per_pass{std::max(desc->property_uint_or_default("samples_per_pass", 64u), 1u)},
          _scheduler{[&] {
              auto s = desc->property_string_or_default(
                  "scheduler", luisa::lazy_construct([&] {
                      return desc->property_string_or_default("scheduler_type", "wavefront");
                  }));
              for (auto &c : s) { c = static_cast<char>(std::tolower(c)); }
              if (s == "wavefront") { return Scheduler::Wavefront; }
              if (s == "persistent") { return Scheduler::Persistent; }
              if (s == "simple") { return Scheduler::Simple; }
              LUISA_ERROR_WITH_LOCATION(
                  "Unknown scheduler type '{}'. "
                  "Supported types are: wavefront, persistent, simple.",
                  s);
          }()} {
        switch (_scheduler) {
            case Scheduler::Simple: break;
            case Scheduler::Wavefront: {
                if (desc->has_property("soa")) { _wavefront_config.global_memory_soa = desc->property_bool("soa"); }
                if (desc->has_property("sort")) { _wavefront_config.gather_by_sorting = desc->property_bool("sort"); }
                if (desc->has_property("compact")) { _wavefront_config.frame_buffer_compaction = desc->property_bool("compact"); }
                if (desc->has_property("instances")) { _wavefront_config.thread_count = std::max<uint>(desc->property_uint("instances"), 1_k); }
                if (desc->has_property("threads")) { _wavefront_config.thread_count = std::max<uint>(desc->property_uint("threads"), 1_k); }
                if (desc->has_property("max_instance_count")) { _wavefront_config.thread_count = std::max<uint>(desc->property_uint("max_instance_count"), 1_k); }
                if (desc->has_property("sort_hints")) { _wavefront_config.hint_fields = desc->property_string_list_or_default("sort_hints"); }
                break;
            }
            case Scheduler::Persistent: {
                _persistent_config = {
                    .thread_count = 128_k,
                    .block_size = 64,
                    .fetch_size = 8,
                    .shared_memory_soa = true,
                    .global_memory_ext = true,
                };
                if (desc->has_property("max_thread_count")) { _persistent_config.thread_count = std::max<uint>(desc->property_uint("max_thread_count"), 5_k); }
                if (desc->has_property("threads")) { _persistent_config.thread_count = std::max<uint>(desc->property_uint("threads"), 5_k); }
                if (desc->has_property("block_size")) { _persistent_config.block_size = std::max<uint>(desc->property_uint("block_size"), 32u); }
                if (desc->has_property("fetch_size")) { _persistent_config.fetch_size = std::max<uint>(desc->property_uint("fetch_size"), 1u); }
                if (desc->has_property("global")) { _persistent_config.global_memory_ext = desc->property_bool("global"); }
                break;
            }
        }
    }
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto samples_per_pass() const noexcept { return _samples_per_pass; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;

    // scheduler config
    [[nodiscard]] auto scheduler() const noexcept { return _scheduler; }
    [[nodiscard]] auto &wavefront_config() const noexcept { return _wavefront_config; }
    [[nodiscard]] auto &persistent_config() const noexcept { return _persistent_config; }
};

class CoroutinePathTracingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        auto spp = camera->node()->spp();
        auto resolution = camera->film()->node()->resolution();
        auto image_file = camera->node()->file();

        auto pixel_count = resolution.x * resolution.y;
        sampler()->reset(command_buffer, resolution, pixel_count, spp);
        command_buffer << synchronize();

        LUISA_INFO(
            "Rendering to '{}' of resolution {}x{} at {}spp.",
            image_file.string(),
            resolution.x, resolution.y, spp);

        using namespace luisa::compute;

        coroutine::Coroutine render = [&](Float time, Float shutter_weight, UInt spp_offset) {
            auto frame_index = spp_offset + dispatch_z();
            auto pixel_id = dispatch_id().xy();
            auto L = Li(camera, frame_index, pixel_id, time);
            camera->film()->accumulate(pixel_id, shutter_weight * L);
        };
        Clock clock_compile;
        auto coro_pt = node<CoroutinePathTracing>();
        auto scheduler = [&]() noexcept -> luisa::unique_ptr<CoroutinePathTracing::SchedulerBase> {
            auto &device = pipeline().device();
            auto &stream = *command_buffer.stream();
            switch (coro_pt->scheduler()) {
                case CoroutinePathTracing::Scheduler::Simple: {
                    coroutine::StateMachineCoroScheduler s{device, render};
                    return luisa::make_unique<decltype(s)>(std::move(s));
                }
                case CoroutinePathTracing::Scheduler::Wavefront: {
                    auto config = coro_pt->wavefront_config();
                    // BUG: metal has trouble with the sorting kernel...
                    if (device.backend_name() == "metal") {
                        config.gather_by_sorting = false;
                        config.hint_fields = {};
                    }
                    if (!config.hint_fields.empty()) {
                        config.hint_range = pipeline().surfaces().size();
                    }
                    LUISA_INFO("config: soa:{}, sort:{}, compact:{}, max_instance_count:{}, hint_range:{}, hint_fields[0]:{}",
                               config.global_memory_soa,
                               config.gather_by_sorting,
                               config.frame_buffer_compaction,
                               config.thread_count,
                               config.hint_range,
                               !config.hint_fields.empty() ? config.hint_fields[0] : "NULL");
                    coroutine::WavefrontCoroScheduler s{device, render, config};
                    return luisa::make_unique<decltype(s)>(std::move(s));
                }
                case CoroutinePathTracing::Scheduler::Persistent: {
                    auto config = coro_pt->persistent_config();
                    LUISA_INFO("config: max_thread_count:{}, block_size:{}, fetch_size:{}, global:{}",
                               config.thread_count,
                               config.block_size,
                               config.fetch_size,
                               config.global_memory_ext);
                    coroutine::PersistentThreadsCoroScheduler s{device, render, config};
                    return luisa::make_unique<decltype(s)>(std::move(s));
                }
                default:
                    break;
            }
            LUISA_ERROR_WITH_LOCATION(
                "Unknown scheduler type '{}'. "
                "Supported types are: wavefront, persistent, simple.",
                luisa::to_string(coro_pt->scheduler()));
        }();
        auto integrator_shader_compilation_time = clock_compile.toc();
        LUISA_INFO("Integrator shader compile in {} ms with {} coroutine scheduler.",
                   integrator_shader_compilation_time,
                   luisa::to_string(coro_pt->scheduler()));
        auto shutter_samples = camera->node()->shutter_samples();
        command_buffer << synchronize();

        LUISA_INFO("Rendering started.");
        Clock clock;
        ProgressBar progress;
        progress.update(0.);
        auto sample_id = 0u;
        for (auto s : shutter_samples) {
            pipeline().update(command_buffer, s.point.time);
            auto aligned_spp = luisa::align(s.spp, coro_pt->samples_per_pass());
            for (auto i = 0u; i < aligned_spp; i += coro_pt->samples_per_pass()) {
                auto ns = std::min<uint>(coro_pt->samples_per_pass(), s.spp - i);
                command_buffer << (*scheduler)(s.point.time, s.point.weight, sample_id)
                                      .dispatch(resolution.x, resolution.y, ns);
                sample_id += ns;
                camera->film()->show(command_buffer);
                auto p = sample_id / static_cast<double>(spp);
                command_buffer << [&progress, p] { progress.update(p); };
            }
        }
        command_buffer << synchronize();
        progress.done();

        auto render_time = clock.toc();
        LUISA_INFO("Rendering finished in {} ms.", render_time);
    }

    [[nodiscard]] Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time) const noexcept override {
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto u_swl = spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d();
        auto sample_wl = [&] { return spectrum->sample(spectrum->node()->is_fixed() ? 0.f : u_swl); };
        auto swl = sample_wl();
        SampledSpectrum beta{swl.dimension(), camera_weight};
        SampledSpectrum Li{swl.dimension()};

        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        $for (depth, node<CoroutinePathTracing>()->max_depth()) {
            $suspend("intersect");
            // trace
            auto hit = pipeline().geometry()->trace_closest(ray);

            // miss
            $if (hit->miss()) {
                if (pipeline().environment()) {
                    $suspend("miss");
                    auto swl = sample_wl();
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                }
                $break;
            };
            auto shape = pipeline().geometry()->instance(hit.inst);
            // hit light
            if (!pipeline().lights().empty()) {
                $if (shape.has_light()) {
                    auto it = pipeline().geometry()->interaction(ray, hit);
                    auto swl = sample_wl();
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                };
            }

            $if (!shape.has_surface()) { $break; };

            $suspend("sample_ray");
            // generate uniform samples
            //$if((pixel_id.x==124) & (pixel_id.y==700)){
            //    device_log("frame_id:{}, ray:o: {}, d:{}, hit:{}, Li:{}, beta:{}", frame_index, ray->origin(), ray->direction(), hit.bary,spectrum->srgb(swl, Li),spectrum->srgb(swl, beta));
            //};
            auto it = pipeline().geometry()->interaction(ray, hit);
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();

            // sample one light
            auto swl = sample_wl();
            auto light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            if (auto pt = node<CoroutinePathTracing>();
                pt->scheduler() == CoroutinePathTracing::Scheduler::Wavefront &&
                !pt->wavefront_config().hint_fields.empty()) {
                $promise("coro_hint", surface_tag);
            }
            $suspend("evaluate_surface");
            swl = sample_wl();
            auto wo = -ray->direction();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<CoroutinePathTracing>()->rr_depth();
            $if (depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
            it = pipeline().geometry()->interaction(ray, hit);
            surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);

            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](const Surface::Closure *closure) noexcept {
                if (auto dispersive = closure->is_dispersive()) {
                    $if (*dispersive) { swl.terminate_secondary(); };
                }
                // direct lighting
                $if (light_sample.eval.pdf > 0.0f & !occluded) {
                    auto wi = light_sample.shadow_ray->direction();
                    auto eval = closure->evaluate(wo, wi);
                    auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                             light_sample.eval.pdf;
                    Li += w * beta * eval.f * light_sample.eval.L;
                };
                // sample material
                auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                ray = it->spawn_ray(surface_sample.wi);
                pdf_bsdf = surface_sample.eval.pdf;
                auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                beta *= w * surface_sample.eval.f;
                // apply eta scale
                auto eta = closure->eta().value_or(1.f);
                $switch (surface_sample.event) {
                    $case (Surface::event_enter) { eta_scale = sqr(eta); };
                    $case (Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                };
            });
            beta = zero_if_any_nan(beta);
            $if (beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
            auto rr_threshold = node<CoroutinePathTracing>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if (depth + 1u >= rr_depth) {
                $if (q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
        return spectrum->srgb(swl, Li);
    }
};

luisa::unique_ptr<Integrator::Instance> CoroutinePathTracing::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<CoroutinePathTracingInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::CoroutinePathTracing)
