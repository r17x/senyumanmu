use cam_vision::LandmarkMesh;

pub struct AUActivations {
    pub au1: f32,
    pub au4: f32,
    pub au6: f32,
    pub au12: f32,
    pub au15: f32,
    pub au17: f32,
}

const ICD_MIN: f32 = 0.05;

const RIGHT_PUPIL: usize = 63;
const LEFT_PUPIL: usize = 64;

const RIGHT_EYEBROW_INNER: usize = 2;
const LEFT_EYEBROW_INNER: usize = 6;
const RIGHT_EYE_INNER: usize = 12;
const LEFT_EYE_INNER: usize = 20;

const RIGHT_EYEBROW_MID: usize = 1;
const LEFT_EYEBROW_MID: usize = 5;
const RIGHT_EYE_OUTER: usize = 8;
const LEFT_EYE_OUTER: usize = 16;

const RIGHT_EYE_START: usize = 8;
const RIGHT_EYE_END: usize = 16;
const LEFT_EYE_START: usize = 16;
const LEFT_EYE_END: usize = 24;

const OUTER_LIPS_LEFT_CORNER: usize = 24;
const OUTER_LIPS_RIGHT_CORNER: usize = 29;
const OUTER_LIPS_UPPER_CENTER_A: usize = 26;
const OUTER_LIPS_UPPER_CENTER_B: usize = 27;
const OUTER_LIPS_LOWER_CENTER_A: usize = 31;
const OUTER_LIPS_LOWER_CENTER_B: usize = 32;

const MEDIAN_LINE_CHIN: usize = 62;

const MIN_POINTS: usize = 65;

fn euclidean(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn eye_height(points: &[(f32, f32)], start: usize, end: usize) -> f32 {
    let slice = &points[start..end];
    let min_y = slice.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let max_y = slice.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
    max_y - min_y
}

fn midpoint_y(points: &[(f32, f32)], a: usize, b: usize) -> f32 {
    (points[a].1 + points[b].1) * 0.5
}

fn compute_au1(points: &[(f32, f32)], icd: f32) -> f32 {
    let left = (points[LEFT_EYE_INNER].1 - points[LEFT_EYEBROW_INNER].1) / icd;
    let right = (points[RIGHT_EYE_INNER].1 - points[RIGHT_EYEBROW_INNER].1) / icd;
    clamp01((left + right) * 0.5)
}

fn compute_au4(points: &[(f32, f32)], icd: f32) -> f32 {
    let left = (points[LEFT_EYEBROW_MID].1 - points[LEFT_EYE_OUTER].1) / icd;
    let right = (points[RIGHT_EYEBROW_MID].1 - points[RIGHT_EYE_OUTER].1) / icd;
    clamp01((left + right) * 0.5)
}

fn compute_au6(points: &[(f32, f32)], icd: f32) -> f32 {
    let left_h = eye_height(points, LEFT_EYE_START, LEFT_EYE_END) / icd;
    let right_h = eye_height(points, RIGHT_EYE_START, RIGHT_EYE_END) / icd;
    let avg_openness = (left_h + right_h) * 0.5;
    clamp01(1.0 - avg_openness)
}

fn compute_au12(points: &[(f32, f32)], icd: f32) -> f32 {
    let upper_center_y = midpoint_y(points, OUTER_LIPS_UPPER_CENTER_A, OUTER_LIPS_UPPER_CENTER_B);
    let lower_center_y = midpoint_y(points, OUTER_LIPS_LOWER_CENTER_A, OUTER_LIPS_LOWER_CENTER_B);
    let center_y = (upper_center_y + lower_center_y) * 0.5;

    let left_corner_y = points[OUTER_LIPS_LEFT_CORNER].1;
    let right_corner_y = points[OUTER_LIPS_RIGHT_CORNER].1;
    let avg_corner_y = (left_corner_y + right_corner_y) * 0.5;

    clamp01((center_y - avg_corner_y) / icd)
}

fn compute_au15(points: &[(f32, f32)], icd: f32) -> f32 {
    let upper_center_y = midpoint_y(points, OUTER_LIPS_UPPER_CENTER_A, OUTER_LIPS_UPPER_CENTER_B);
    let lower_center_y = midpoint_y(points, OUTER_LIPS_LOWER_CENTER_A, OUTER_LIPS_LOWER_CENTER_B);
    let center_y = (upper_center_y + lower_center_y) * 0.5;

    let left_corner_y = points[OUTER_LIPS_LEFT_CORNER].1;
    let right_corner_y = points[OUTER_LIPS_RIGHT_CORNER].1;
    let avg_corner_y = (left_corner_y + right_corner_y) * 0.5;

    clamp01((avg_corner_y - center_y) / icd)
}

fn compute_au17(points: &[(f32, f32)], icd: f32) -> f32 {
    let lower_lip_y = midpoint_y(points, OUTER_LIPS_LOWER_CENTER_A, OUTER_LIPS_LOWER_CENTER_B);
    let chin_y = points[MEDIAN_LINE_CHIN].1;
    clamp01((chin_y - lower_lip_y) / icd)
}

const THRESHOLD_AU1: f32 = 0.22;
const THRESHOLD_AU4: f32 = 0.08;
const THRESHOLD_AU6: f32 = 0.25;
const THRESHOLD_AU12: f32 = 0.04;
const THRESHOLD_AU15: f32 = 0.03;

pub enum AffectLabel {
    Happy,
    Sad,
    Neutral,
}

impl std::fmt::Display for AffectLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AffectLabel::Happy => write!(f, "happy"),
            AffectLabel::Sad => write!(f, "sad"),
            AffectLabel::Neutral => write!(f, "neutral"),
        }
    }
}

pub fn classify_affect(au: &AUActivations) -> AffectLabel {
    let happy = au.au6 > THRESHOLD_AU6 && au.au12 > THRESHOLD_AU12;

    let sad_count = (au.au1 > THRESHOLD_AU1) as u8
        + (au.au4 > THRESHOLD_AU4) as u8
        + (au.au15 > THRESHOLD_AU15) as u8;
    let sad = sad_count >= 2;

    if happy {
        AffectLabel::Happy
    } else if sad {
        AffectLabel::Sad
    } else {
        AffectLabel::Neutral
    }
}

pub fn compute_au_activations(mesh: &LandmarkMesh) -> AUActivations {
    let zero = AUActivations {
        au1: 0.0,
        au4: 0.0,
        au6: 0.0,
        au12: 0.0,
        au15: 0.0,
        au17: 0.0,
    };

    if mesh.points.len() < MIN_POINTS {
        return zero;
    }

    let icd = euclidean(mesh.points[RIGHT_PUPIL], mesh.points[LEFT_PUPIL]);
    if icd < ICD_MIN {
        return zero;
    }

    let pts = &mesh.points;

    AUActivations {
        au1: compute_au1(pts, icd),
        au4: compute_au4(pts, icd),
        au6: compute_au6(pts, icd),
        au12: compute_au12(pts, icd),
        au15: compute_au15(pts, icd),
        au17: compute_au17(pts, icd),
    }
}
