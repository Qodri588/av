#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform vec2 u_inv_resolution;

float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec3 rgb_m  = texture(u_tex, v_uv).rgb;
    vec3 rgb_nw = texture(u_tex, v_uv + vec2(-1.0,  1.0) * u_inv_resolution).rgb;
    vec3 rgb_ne = texture(u_tex, v_uv + vec2( 1.0,  1.0) * u_inv_resolution).rgb;
    vec3 rgb_sw = texture(u_tex, v_uv + vec2(-1.0, -1.0) * u_inv_resolution).rgb;
    vec3 rgb_se = texture(u_tex, v_uv + vec2( 1.0, -1.0) * u_inv_resolution).rgb;

    float luma_m  = luma(rgb_m);
    float luma_nw = luma(rgb_nw);
    float luma_ne = luma(rgb_ne);
    float luma_sw = luma(rgb_sw);
    float luma_se = luma(rgb_se);
    float luma_min = min(luma_m, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    float luma_max = max(luma_m, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));

    vec2 direction;
    direction.x = -((luma_nw + luma_ne) - (luma_sw + luma_se));
    direction.y =  ((luma_nw + luma_sw) - (luma_ne + luma_se));

    float reduce = max((luma_nw + luma_ne + luma_sw + luma_se) * 0.03125, 0.0078125);
    float inverse_min = 1.0 / (min(abs(direction.x), abs(direction.y)) + reduce);
    direction = clamp(direction * inverse_min, vec2(-8.0), vec2(8.0)) * u_inv_resolution;

    vec3 rgb_a = 0.5 * (
        texture(u_tex, v_uv + direction * (1.0 / 3.0 - 0.5)).rgb +
        texture(u_tex, v_uv + direction * (2.0 / 3.0 - 0.5)).rgb);
    vec3 rgb_b = rgb_a * 0.5 + 0.25 * (
        texture(u_tex, v_uv + direction * -0.5).rgb +
        texture(u_tex, v_uv + direction *  0.5).rgb);

    float luma_b = luma(rgb_b);
    fragColor = vec4((luma_b < luma_min || luma_b > luma_max) ? rgb_a : rgb_b, 1.0);
}
