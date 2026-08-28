#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform int   u_has_image;
uniform float u_opacity;
uniform float u_img_aspect;     // image width / height
uniform float u_canvas_aspect;  // canvas width / height
uniform float u_zoom;           // 1.0 + reactive zoom
uniform int   u_fit_mode;        // 0=fit, 1=cover, 2=manual crop
uniform float u_crop_zoom;
uniform vec2  u_crop_offset;

void main() {
    if (u_has_image == 0) {
        fragColor = vec4(0.0, 0.0, 0.0, u_opacity);
        return;
    }

    vec2 uv = v_uv - 0.5;
    bool outside = false;
    if (u_fit_mode == 0) {
        if (u_canvas_aspect > u_img_aspect) {
            float width = u_img_aspect / u_canvas_aspect;
            outside = abs(uv.x) > width * 0.5;
            uv.x /= width;
        } else {
            float height = u_canvas_aspect / u_img_aspect;
            outside = abs(uv.y) > height * 0.5;
            uv.y /= height;
        }
    } else {
        if (u_canvas_aspect > u_img_aspect) {
            uv.y *= u_img_aspect / u_canvas_aspect;
        } else {
            uv.x *= u_canvas_aspect / u_img_aspect;
        }
        if (u_fit_mode == 2) {
            uv = uv / max(u_crop_zoom, 0.01) + u_crop_offset;
        }
    }
    uv /= max(u_zoom, 0.0001);   // reactive zoom-in
    uv += 0.5;

    if (outside) {
        fragColor = vec4(0.0, 0.0, 0.0, u_opacity);
        return;
    }
    vec3 c = texture(u_tex, clamp(uv, 0.0, 1.0)).rgb;
    fragColor = vec4(c, u_opacity);
}
