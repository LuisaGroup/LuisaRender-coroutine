import numpy as np
from sys import argv
from os import environ


def rgb2srgb(image):
    return np.uint8(np.round(np.clip(np.where(
        image <= 0.00304,
        12.92 * image,
        1.055 * np.power(np.maximum(image, 0), 1.0 / 2.4) - 0.055
    ) * 255, 0, 255)))


def tonemapping(x):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return rgb2srgb(x * (a * x + b) / (x * (c * x + d) + e))


"""
The fitted AgX implementation is from https://iolite-engine.com/blog_posts/minimal_agx_implementation

// MIT License
//
// Copyright (c) 2024 Missing Deadlines (Benjamin Wrensch)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// All values used to derive this implementation are sourced from Troy’s initial AgX implementation/OCIO config file available here:
//   https://github.com/sobotka/AgX

// 0: Default, 1: Golden, 2: Punchy
#define AGX_LOOK 0

// Mean error^2: 3.6705141e-06
vec3 agxDefaultContrastApprox(vec3 x) {
  vec3 x2 = x * x;
  vec3 x4 = x2 * x2;
  
  return + 15.5     * x4 * x2
         - 40.14    * x4 * x
         + 31.96    * x4
         - 6.868    * x2 * x
         + 0.4298   * x2
         + 0.1191   * x
         - 0.00232;
}

vec3 agx(vec3 val) {
  const mat3 agx_mat = mat3(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992,  0.878468636469772,  0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    
  const float min_ev = -12.47393f;
  const float max_ev = 4.026069f;

  // Input transform (inset)
  val = agx_mat * val;
  
  // Log2 space encoding
  val = clamp(log2(val), min_ev, max_ev);
  val = (val - min_ev) / (max_ev - min_ev);
  
  // Apply sigmoid function approximation
  val = agxDefaultContrastApprox(val);

  return val;
}

vec3 agxEotf(vec3 val) {
  const mat3 agx_mat_inv = mat3(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    
  // Inverse input transform (outset)
  val = agx_mat_inv * val;
  
  // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
  // NOTE: We're linearizing the output here. Comment/adjust when
  // *not* using a sRGB render target
  val = pow(val, vec3(2.2));

  return val;
}

vec3 agxLook(vec3 val) {
  const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
  float luma = dot(val, lw);
  
  // Default
  vec3 offset = vec3(0.0);
  vec3 slope = vec3(1.0);
  vec3 power = vec3(1.0);
  float sat = 1.0;
 
#if AGX_LOOK == 1
  // Golden
  slope = vec3(1.0, 0.9, 0.5);
  power = vec3(0.8);
  sat = 0.8;
#elif AGX_LOOK == 2
  // Punchy
  slope = vec3(1.0);
  power = vec3(1.35, 1.35, 1.35);
  sat = 1.4;
#endif
  
  // ASC CDL
  val = pow(val * slope + offset, power);
  return luma + sat * (val - luma);
}

// Sample usage
void main(/*...*/) {
  // ...
  value = agx(value);
  value = agxLook(value); // Optional
  value = agxEotf(value);
  // ...
}

// 7th Order Polynomial Approximation: Mean error^2: 1.85907662e-06
vec3 agxDefaultContrastApprox(vec3 x) {
  vec3 x2 = x * x;
  vec3 x4 = x2 * x2;
  vec3 x6 = x4 * x2;
  
  return - 17.86     * x6 * x
         + 78.01     * x6
         - 126.7     * x4 * x
         + 92.06     * x4
         - 28.72     * x2 * x
         + 4.361     * x2
         - 0.1718    * x
         + 0.002857;
}
"""


def agx_default_contrast_approx(x: np.ndarray):
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    return (-17.86 * x6 * x + 78.01 * x6
            - 126.7 * x4 * x + 92.06 * x4
            - 28.72 * x2 * x + 4.361 * x2
            - 0.1718 * x + 0.002857)


def agx(x: np.ndarray):
    agx_mat = np.array([
        [0.842479062253094, 0.0423282422610123, 0.0423756549057051],
        [0.0784335999999992, 0.878468636469772, 0.0784336],
        [0.0792237451477643, 0.0791661274605434, 0.879142973793104]
    ])
    min_ev = -12.47393
    max_ev = 4.026069
    x = np.maximum(np.matmul(x, agx_mat), np.power(2, min_ev - 1))
    x = np.clip(np.log2(x), min_ev, max_ev)
    x = (x - min_ev) / (max_ev - min_ev)
    return agx_default_contrast_approx(x)


def agx_eotf(x: np.ndarray):
    agx_mat_inv = np.array([
        [1.19687900512017, -0.0528968517574562, -0.0529716355144438],
        [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
        [-0.0990297440797205, -0.0989611768448433, 1.15107367264116]
    ])
    return np.power(np.maximum(np.matmul(x, agx_mat_inv), 0), 2.2)


def agx_look(x: np.ndarray, look="default"):
    lw = np.array([0.2126, 0.7152, 0.0722])
    luma = np.dot(x, lw)[..., None]
    offset = np.array([0.0, 0.0, 0.0])
    slope = np.array([1.0, 1.0, 1.0])
    power = np.array([1.0, 1.0, 1.0])
    sat = 1.0
    if look == "golden":
        slope = np.array([1.0, 0.9, 0.5])
        power = np.array([0.8, 0.8, 0.8])
        sat = 0.8
    elif look == "punchy":
        slope = np.array([1.0, 1.0, 1.0])
        power = np.array([1.35, 1.35, 1.35])
        sat = 1.4
    x = np.power(x * slope + offset, power)
    return luma + sat * (x - luma)


def agx_tonemap(x: np.ndarray, look="default"):
    return rgb2srgb(agx_eotf(agx_look(agx(x), look)))


def imread(filename):
    return np.maximum(
        np.nan_to_num(cv.imread(filename, cv.IMREAD_UNCHANGED)[:, :, :3], nan=0.0, posinf=1e3, neginf=0), 0.0)[..., ::-1]


if __name__ == "__main__":
    environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2 as cv

    filename = argv[1]
    exp = 0 if len(argv) == 2 else float(argv[2])
    look = "default" if len(argv) <= 3 else argv[3].lower()
    assert filename.endswith(".exr")
    image = imread(filename) * (2 ** exp)
    cv.imwrite(f"{filename[:-4]}.png", agx_tonemap(image, look)[..., ::-1])
