//
// Created by Mike Smith on 2022/1/25.
//

#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

Texture::Texture(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::TEXTURE} {}

uint Texture::handle_tag() const noexcept {
    static std::mutex mutex;
    static luisa::unordered_map<luisa::string, uint, Hash64> impl_to_tag;
    std::scoped_lock lock{mutex};
    auto [iter, _] = impl_to_tag.try_emplace(
        luisa::string{impl_type()},
        static_cast<uint>(impl_to_tag.size()));
    return iter->second;
}

TextureHandle TextureHandle::encode_constant(uint tag, float3 v, float alpha) noexcept {
    if (tag > tag_mask) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid tag for texture handle: {}.", tag);
    }
    if (alpha < 0.0f || alpha > fixed_point_alpha_max) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid alpha for texture handle: {}. "
            "Clamping to [0, {}].",
            alpha, fixed_point_alpha_max);
    }
    auto fp_alpha = static_cast<uint>(std::round(
                        std::clamp(alpha, 0.0f, fixed_point_alpha_max) *
                        fixed_point_alpha_scale))
                    << texture_id_offset_shift;
    return {.id_and_tag = tag | fp_alpha, .compressed_v = {v.x, v.y, v.z}};
}

TextureHandle TextureHandle::encode_texture(uint tag, uint tex_id, float3 v) noexcept {
    if (tag > tag_mask) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid tag for texture handle: {}.", tag);
    }
    if (tex_id > (~0u >> texture_id_offset_shift)) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid id for texture handle: {}.", tex_id);
    }
    return {.id_and_tag = tag | tex_id, .compressed_v = {v.x, v.y, v.z}};
}

ImageTexture::ImageTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
    : Texture{scene, desc} {
    auto address = desc->property_string_or_default("address_mode", "repeat");
    auto filter = desc->property_string_or_default("filter_mode", "bilinear");
    auto address_mode = [&address, desc] {
        for (auto &c : address) { c = static_cast<char>(tolower(c)); }
        if (address == "zero") { return TextureSampler::Address::ZERO; }
        if (address == "edge") { return TextureSampler::Address::EDGE; }
        if (address == "mirror") { return TextureSampler::Address::MIRROR; }
        if (address == "repeat") { return TextureSampler::Address::REPEAT; }
        LUISA_ERROR(
            "Invalid texture address mode '{}'. [{}]",
            address, desc->source_location().string());
    }();
    auto filter_mode = [&filter, desc] {
        for (auto &c : filter) { c = static_cast<char>(tolower(c)); }
        if (filter == "point") { return TextureSampler::Filter::POINT; }
        if (filter == "bilinear") { return TextureSampler::Filter::BILINEAR; }
        if (filter == "trilinear") { return TextureSampler::Filter::TRILINEAR; }
        if (filter == "anisotropic" || filter == "aniso") { return TextureSampler::Filter::ANISOTROPIC; }
        LUISA_ERROR(
            "Invalid texture filter mode '{}'. [{}]",
            filter, desc->source_location().string());
    }();
    _sampler = {filter_mode, address_mode};
}

}// namespace luisa::render