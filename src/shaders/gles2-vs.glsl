#version 100

attribute vec2 a_position;
attribute vec2 a_texcoord;
attribute vec4 a_color;
uniform mat4 u_mvp;

varying vec2 v_uv;
varying vec4 v_color;

void main()
{
    v_uv = a_texcoord;
    v_color = a_color;
    gl_Position = u_mvp * vec4(a_position.xy, 0, 1);
}
