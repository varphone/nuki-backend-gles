use gls::{gl, prelude::Bindable, GLfloat, GLint, GLsizei, GLsizeiptr, GLubyte, GLuint};
use nuki::{
    Allocator, AntiAliasing, Buffer, Context, ConvertConfig, DrawNullTexture,
    DrawVertexLayoutAttribute, DrawVertexLayoutElements, DrawVertexLayoutFormat, FontAtlas,
    FontAtlasFormat, Handle, Rect,
};
use std::rc::Rc;

macro_rules! offset_of {
    ($t:ty, $field:ident) => {{
        const X: *const $t = ::std::mem::align_of::<$t>() as *const $t;
        unsafe { (&(*X).$field as *const _ as usize) - (X as usize) }
    }};
}

#[derive(Clone, Default)]
struct RenderState<'a> {
    vbo: gls::Buffer,
    ebo: gls::Buffer,
    prog: gls::Program,
    font_texs: Vec<Rc<gls::Texture<'a>>>,
    position_aloc: GLint,
    texcoord_aloc: GLint,
    color_aloc: GLint,
    mvp_uloc: GLint,
    texture_uloc: GLint,
    vs: GLsizei,
    vp: GLsizei,
    vt: GLsizei,
    vc: GLsizei,
}

impl<'a> RenderState<'a> {
    pub fn new(max_vertex_buffer: usize, max_element_buffer: usize) -> Self {
        let mut state: Self = Default::default();

        state.vbo = gls::Buffer::new_array();
        state.vbo.stream_draw_data_null::<u8>(max_vertex_buffer);
        state.ebo = gls::Buffer::new_element_array();
        state.ebo.stream_draw_data_null::<u8>(max_element_buffer);

        state.prog = gls::Program::from_sources(&[
            (include_str!("shaders/gles2-fs.glsl"), gl::FRAGMENT_SHADER),
            (include_str!("shaders/gles2-vs.glsl"), gl::VERTEX_SHADER),
        ])
        .unwrap();

        state.position_aloc = state.prog.locate_attrib("a_position").unwrap_or(-1);
        state.texcoord_aloc = state.prog.locate_attrib("a_texcoord").unwrap_or(-1);
        state.color_aloc = state.prog.locate_attrib("a_color").unwrap_or(-1);
        state.mvp_uloc = state.prog.locate_uniform("u_mvp").unwrap_or(-1);
        state.texture_uloc = state.prog.locate_uniform("u_texture").unwrap_or(-1);

        state.vs = std::mem::size_of::<Vertex>() as GLsizei;
        state.vp = offset_of!(Vertex, position) as GLsizei;
        state.vt = offset_of!(Vertex, uv) as GLsizei;
        state.vc = offset_of!(Vertex, col) as GLsizei;

        state
    }

    pub fn add_font_texture<'b>(&mut self, image: &'b [u8], width: u32, height: u32) -> Handle {
        let tex = gls::TextureLoader::default()
            .with_bytes(image)
            .with_size(width as usize, height as usize)
            .with_internal_format(gls::TextureFormat::Rgba)
            .with_format(gls::TextureFormat::Rgba)
            .with_linear()
            .load()
            .unwrap();
        let handle = Handle::from_id(tex.id() as i32);
        self.font_texs.push(Rc::new(tex));
        handle
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct Vertex {
    position: [GLfloat; 2],
    uv: [GLfloat; 2],
    col: [GLubyte; 4],
}

/// Options to control the drawing.
#[derive(Clone, Copy, Default, Debug)]
pub struct DrawOptions {
    display_size: (usize, usize),
    dpi_factor: (f32, f32),
    scale_factor: (f32, f32),
    viewport: (isize, isize, isize, isize),
}

impl DrawOptions {
    /// Create a new DrawOptions.
    ///
    /// # Parameters
    ///
    /// * `width` - The width of the displayer.
    /// * `height` - The height of the displayer.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            display_size: (width, height),
            dpi_factor: (1.0, 1.0),
            scale_factor: (1.0, 1.0),
            viewport: (0, 0, width as isize, height as isize),
        }
    }

    /// Change the DPI factor.
    pub fn with_dpi_factor(mut self, x: f32, y: f32) -> Self {
        self.dpi_factor = (x, y);
        self
    }

    // Change the scale factor.
    pub fn with_scale_factor(mut self, x: f32, y: f32) -> Self {
        self.scale_factor = (x, y);
        self
    }
}

#[derive(Clone)]
pub struct Drawer<'a> {
    alloc: Allocator,
    cmds: Buffer,
    vbuf: Buffer,
    ebuf: Buffer,
    config: ConvertConfig,
    vertex_layout: DrawVertexLayoutElements,
    null: DrawNullTexture,
    state: RenderState<'a>,
}

impl<'a> Drawer<'a> {
    pub fn new(alloc: Allocator, max_vertex_buffer: usize, max_element_buffer: usize) -> Self {
        let vertex_layout = DrawVertexLayoutElements::new(&[
            (
                DrawVertexLayoutAttribute::Position,
                DrawVertexLayoutFormat::Float,
                unsafe { &(*(::std::ptr::null::<Vertex>())).position as *const _ as usize },
            ),
            (
                DrawVertexLayoutAttribute::TexCoord,
                DrawVertexLayoutFormat::Float,
                unsafe { &(*(::std::ptr::null::<Vertex>())).uv as *const _ as usize },
            ),
            (
                DrawVertexLayoutAttribute::Color,
                DrawVertexLayoutFormat::R8G8B8A8,
                unsafe { &(*(::std::ptr::null::<Vertex>())).col as *const _ as usize },
            ),
            (
                DrawVertexLayoutAttribute::AttributeCount,
                DrawVertexLayoutFormat::Count,
                0,
            ),
        ]);
        let mut config: ConvertConfig = Default::default();
        config.set_global_alpha(1.0);
        config.set_line_aa(AntiAliasing::On);
        config.set_shape_aa(AntiAliasing::On);
        config.set_circle_segment_count(22);
        config.set_arc_segment_count(22);
        config.set_curve_segment_count(22);
        //config.set_null(DrawNullTexture::default());
        config.set_vertex_layout(&vertex_layout);
        config.set_vertex_size(std::mem::size_of::<Vertex>());
        Self {
            cmds: Buffer::new(&alloc),
            vbuf: Buffer::with_size(&alloc, max_vertex_buffer),
            ebuf: Buffer::with_size(&alloc, max_element_buffer),
            alloc,
            config,
            vertex_layout,
            null: Default::default(),
            state: RenderState::new(max_vertex_buffer, max_element_buffer),
        }
    }

    /// Draw all elements in the context.
    pub fn draw(&mut self, ctx: &mut Context, options: &DrawOptions) {
        gls::enable(gl::BLEND);
        gls::blend_equation(gl::FUNC_ADD);
        gls::blend_func(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        gls::disable(gl::CULL_FACE);
        gls::disable(gl::DEPTH_TEST);
        gls::enable(gl::SCISSOR_TEST);
        gls::active_texture(gl::TEXTURE0);

        // Setup program
        let mvp = self.get_projection(options);
        self.state.prog.bind();
        self.state
            .prog
            .set_uniform(self.state.mvp_uloc, gls::uniform!(mat4(&mvp)));
        self.state
            .prog
            .set_uniform(self.state.texture_uloc, gls::uniform!(int(0)));

        self.state.vbo.bind();
        self.state.ebo.bind();

        let a_position = gls::VertexAttrib::new(
            self.state.position_aloc as GLuint,
            2,
            gl::FLOAT,
            gl::FALSE,
            self.state.vs,
            self.state.vp as GLsizeiptr,
        );

        let a_texcoord = gls::VertexAttrib::new(
            self.state.texcoord_aloc as GLuint,
            2,
            gl::FLOAT,
            gl::FALSE,
            self.state.vs,
            self.state.vt as GLsizeiptr,
        );

        let a_color = gls::VertexAttrib::new(
            self.state.color_aloc as GLuint,
            4,
            gl::UNSIGNED_BYTE,
            gl::TRUE,
            self.state.vs,
            self.state.vc as GLsizeiptr,
        );

        a_position.bind();
        a_texcoord.bind();
        a_color.bind();

        self.cmds.clear();
        self.vbuf.clear();
        self.ebuf.clear();
        self.config.set_null(self.null.clone());

        ctx.convert(&mut self.cmds, &mut self.vbuf, &mut self.ebuf, &self.config);

        self.state.vbo.update(self.vbuf.as_bytes());
        self.state.ebo.update(self.ebuf.as_bytes());

        let mut eptr: *mut u16 = std::ptr::null_mut();
        for cmd in ctx.draw_command_iterator(&self.cmds) {
            if cmd.elem_count() < 1 {
                continue;
            }

            let count = cmd.elem_count();
            let id = cmd.texture().id().unwrap();
            self.clip_rect(cmd.clip_rect(), options);
            gls::bind_texture(gl::TEXTURE_2D, id as GLuint);
            gls::draw_elements(
                gl::TRIANGLES,
                count as GLsizei,
                gl::UNSIGNED_SHORT,
                eptr as GLsizeiptr,
            );
            eptr = unsafe { eptr.add(count as usize) };
        }

        gls::disable(gl::BLEND);
        gls::enable(gl::CULL_FACE);
        gls::enable(gl::DEPTH_TEST);
        gls::disable(gl::SCISSOR_TEST);
    }

    pub fn add_font_texture<'b>(&mut self, data: &'b [u8], width: u32, height: u32) -> Handle {
        self.state.add_font_texture(data, width, height)
    }

    pub fn bake_font_atlas<'b>(&mut self, atlas: &'b mut FontAtlas) -> Handle {
        let (image, w, h) = atlas.bake(FontAtlasFormat::Rgba32);
        let handle = self.add_font_texture(image, w, h);
        atlas.end(handle, Some(&mut self.null));
        handle
    }

    #[inline]
    pub fn clip_rect(&self, rect: &Rect, options: &DrawOptions) {
        let dw = options.display_size.0 as f32;
        let dh = options.display_size.1 as f32;
        if (rect.x - -8192.0).abs() < f32::EPSILON
            && (rect.y - -8192.0).abs() < f32::EPSILON
            && (rect.w - 16384.0).abs() < f32::EPSILON
            && (rect.h - 16384.0).abs() < f32::EPSILON
        {
            gls::scissor(0, 0, dw as GLsizei, dh as GLsizei);
        } else {
            let fx = options.dpi_factor.0 * options.scale_factor.0;
            let fy = options.dpi_factor.1 * options.scale_factor.1;
            let x = (rect.x * fx).floor() as GLint;
            let w = (rect.w * fx).ceil() as GLsizei;
            let h = (rect.h * fy).ceil() as GLsizei;
            let y = (dh - rect.y - h as f32).floor() as GLint;
            gls::scissor(x, y, w, h);
        }
    }

    pub fn get_projection(&self, options: &DrawOptions) -> gls::Matrix4 {
        let fx = options.dpi_factor.0 / options.scale_factor.0;
        let fy = options.dpi_factor.1 / options.scale_factor.1;
        let w = (options.display_size.0 as f32 * fx).ceil();
        let h = (options.display_size.1 as f32 * fy).ceil();
        gls::Matrix4::new_orthographic(0.0, w, h, 0.0, -1.0, 1.0)
    }

    pub fn get_null(&self) -> &DrawNullTexture {
        &self.null
    }

    pub fn get_null_mut(&mut self) -> &mut DrawNullTexture {
        &mut self.null
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
