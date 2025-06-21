use nlm_impl::{Array_F32_1D, Config, Context};

pub struct NlmContext<B: nlm_impl::backends::Backend> {
    context: Context<B>,
}

impl<B: nlm_impl::backends::Backend> NlmContext<B> {
    pub fn new() -> Self {
        let config: Config<B> = Config::<B>::new();
        let context: Context<B> = Context::new(config);
        NlmContext { context }
    }

    pub fn with_config(config: Config<B>) -> Self {
        NlmContext {
            context: Context::new(config),
        }
    }
}

impl<B: nlm_impl::backends::Backend> NlmContext<B> {
    pub fn patchwise(
        &self,
        flat_channels: &[f32],
        (dim_x, dim_y): (i64, i64),
        patch_side: i64,
        search_window_size: i64,
        h: f32,
    ) -> Vec<f32> {
        let entry_call: Array_F32_1D<'_, B> = 
        {
            let futhark_input: Array_F32_1D<'_, B> =
                Array_F32_1D::new(&self.context, flat_channels, flat_channels.len());
            self.context
                .entry_patchwise_nlm(
                    &futhark_input,
                    dim_x,
                    dim_y,
                    patch_side,
                    search_window_size,
                    h,
                )
                .unwrap()
        };

        assert_eq!(Context::sync(&self.context), true);

        let mut vec: Vec<f32> = Vec::new();
        let _ = entry_call.values(&mut vec);

        vec
    }
}

