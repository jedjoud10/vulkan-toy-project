use ash::vk;

pub struct Samplers {
    pub linear: vk::Sampler,
    pub nearest: vk::Sampler,
}

impl Samplers {
    pub unsafe fn create_samplers(device: &ash::Device) -> Self {
        let linear_create_info = vk::SamplerCreateInfo::default()
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_filter(vk::Filter::LINEAR)
            .mag_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .max_lod(100f32)
            .min_lod(0f32);
        let linear = device.create_sampler(&linear_create_info, None).unwrap();

        let nearest_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .max_lod(100f32)
            .min_lod(0f32);

        let nearest = device.create_sampler(&nearest_create_info, None).unwrap();

        Self {
            linear,
            nearest,
        }
    }

    pub unsafe fn destroy_samplers(self, device: &ash::Device) {
        device.destroy_sampler(self.linear, None);
        device.destroy_sampler(self.nearest, None);
    }
}