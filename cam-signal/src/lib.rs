pub struct ColorSample {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub frame_index: u32,
}

pub struct FrameSignals {
    pub frame_index: u32,
    pub spatial_means: [f32; 3],
    pub color_sample: ColorSample,
}

pub struct SignalStream {
    pub samples: Vec<FrameSignals>,
}

impl SignalStream {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn push(&mut self, s: FrameSignals) {
        self.samples.push(s);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl Default for SignalStream {
    fn default() -> Self {
        Self::new()
    }
}
