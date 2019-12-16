use gl::{GLfloat, GLint, GLsizei, GLsizeiptr, GLubyte, GLuint};
use nuki::{
    Allocator, AntiAliasing, Buffer, Context, ConvertConfig, DrawNullTexture,
    DrawVertexLayoutAttribute, DrawVertexLayoutElements, DrawVertexLayoutFormat, FontAtlas,
    FontAtlasFormat, Handle, Rect, Vec2, Vec2i,
};

#[derive(Clone, Default)]
struct RenderState {
    vbo: GLuint,
    ebo: GLuint,
    prog: GLuint,
    vert_shdr: GLuint,
    frag_shdr: GLuint,
    attrib_pos: GLint,
    attrib_uv: GLint,
    attrib_col: GLint,
    uniform_tex: GLint,
    uniform_proj: GLint,
    font_texs: Vec<GLuint>,
    vs: GLsizei,
    vp: GLsizei,
    vt: GLsizei,
    vc: GLsizei,
}

impl RenderState {
    pub fn new(alloc: &mut Allocator) -> Self {
        let mut state: Self = Default::default();
        // Create shader program
        state.prog = gl::create_program().unwrap();
        state.vert_shdr = gl::create_shader(gl::VERTEX_SHADER).unwrap();
        state.frag_shdr = gl::create_shader(gl::FRAGMENT_SHADER).unwrap();
        let vertex_shader = include_bytes!("shaders/gles2-vs.glsl");
        gl::shader_source(
            state.vert_shdr,
            String::from_utf8(vertex_shader.to_vec()).unwrap(),
        );
        let fragment_shader = include_bytes!("shaders/gles2-fs.glsl");
        gl::shader_source(
            state.frag_shdr,
            String::from_utf8(fragment_shader.to_vec()).unwrap(),
        );
        gl::compile_shader(state.vert_shdr);
        assert_eq!(
            gl::get_shaderiv(state.vert_shdr, gl::COMPILE_STATUS),
            gl::TRUE as GLint
        );
        gl::compile_shader(state.frag_shdr);
        assert_eq!(
            gl::get_shaderiv(state.frag_shdr, gl::COMPILE_STATUS),
            gl::TRUE as GLint
        );
        gl::attach_shader(state.prog, state.vert_shdr);
        gl::attach_shader(state.prog, state.frag_shdr);
        gl::link_program(state.prog);
        assert_eq!(
            gl::get_programiv(state.prog, gl::LINK_STATUS),
            gl::TRUE as GLint
        );

        state.uniform_tex = gl::get_uniform_location(state.prog, "Texture").unwrap_or(-1);
        state.uniform_proj = gl::get_uniform_location(state.prog, "ProjMtx").unwrap_or(-1);
        state.attrib_pos = gl::get_attrib_location(state.prog, "Position").unwrap_or(-1);
        state.attrib_uv = gl::get_attrib_location(state.prog, "TexCoord").unwrap_or(-1);
        state.attrib_col = gl::get_attrib_location(state.prog, "Color").unwrap_or(-1);

        state.vs = std::mem::size_of::<Vertex>() as GLsizei;
        state.vp = unsafe { &(*(::std::ptr::null::<Vertex>())).position as *const _ as GLsizei };
        state.vt = unsafe { &(*(::std::ptr::null::<Vertex>())).uv as *const _ as GLsizei };
        state.vc = unsafe { &(*(::std::ptr::null::<Vertex>())).col as *const _ as GLsizei };

        // Allocate the buffers
        let mut buffers: [GLuint; 2] = [0, 0];
        gl::gen_buffers(&mut buffers);
        state.vbo = buffers[0];
        state.ebo = buffers[1];

        gl::bind_texture(gl::TEXTURE_2D, 0);
        gl::bind_buffer(gl::ARRAY_BUFFER, 0);
        gl::bind_buffer(gl::ELEMENT_ARRAY_BUFFER, 0);

        state
    }

    pub fn add_font_texture(&mut self, image: &[u8], width: u32, height: u32) -> Handle {
        let mut tex: GLuint = 0;
        gl::gen_textures(unsafe { std::mem::transmute::<&mut GLuint, &mut [GLuint; 1]>(&mut tex) });
        gl::bind_texture(gl::TEXTURE_2D, tex);
        gl::tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as GLint);
        gl::tex_parameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as GLint);
        gl::tex_image2d(
            gl::TEXTURE_2D,
            0,
            gl::RGBA as GLint,
            width as GLsizei,
            height as GLsizei,
            0,
            gl::RGBA,
            gl::UNSIGNED_BYTE,
            Some(image),
        );
        gl::bind_texture(gl::TEXTURE_2D, 0);
        self.font_texs.push(tex);
        Handle::from_id(tex as i32)
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

#[derive(Clone, Default)]
pub struct Drawer {
    cmds: Buffer,
    vbuf: Buffer,
    ebuf: Buffer,
    config: ConvertConfig,
    vertex_layout: DrawVertexLayoutElements,
    null: DrawNullTexture,
    state: RenderState,
}

impl Drawer {
    pub fn new(alloc: &mut Allocator, max_vertex_buffer: usize, max_element_buffer: usize) -> Self {
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
            cmds: Buffer::new(alloc),
            vbuf: Buffer::with_size(alloc, max_vertex_buffer),
            ebuf: Buffer::with_size(alloc, max_element_buffer),
            config: config,
            vertex_layout: vertex_layout,
            null: Default::default(),
            state: RenderState::new(alloc),
        }
    }

    /// Draw all elements in the context.
    pub fn draw(&mut self, ctx: &mut Context, options: &DrawOptions) {
        gl::enable(gl::BLEND);
        gl::blend_equation(gl::FUNC_ADD);
        gl::blend_func(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        gl::disable(gl::CULL_FACE);
        gl::disable(gl::DEPTH_TEST);
        gl::enable(gl::SCISSOR_TEST);
        gl::active_texture(gl::TEXTURE0);

        // Setup program
        gl::use_program(self.state.prog);
        gl::uniform1i(self.state.uniform_tex, 0);
        let mvp = self.ortho(options);
        gl::uniform_matrix4fv(self.state.uniform_proj, gl::FALSE, &mvp);

        // Bind buffers
        gl::bind_buffer(gl::ARRAY_BUFFER, self.state.vbo);
        gl::bind_buffer(gl::ELEMENT_ARRAY_BUFFER, self.state.ebo);

        // Buffer setup
        gl::enable_vertex_attrib_array(self.state.attrib_pos as GLuint);
        gl::enable_vertex_attrib_array(self.state.attrib_uv as GLuint);
        gl::enable_vertex_attrib_array(self.state.attrib_col as GLuint);

        let a: gl::GLsizeiptr = 0;

        gl::vertex_attrib_pointer(
            self.state.attrib_pos as GLuint,
            2,
            gl::FLOAT,
            gl::FALSE,
            self.state.vs,
            self.state.vp as GLsizeiptr,
        );
        gl::vertex_attrib_pointer(
            self.state.attrib_uv as GLuint,
            2,
            gl::FLOAT,
            gl::FALSE,
            self.state.vs,
            self.state.vt as GLsizeiptr,
        );
        gl::vertex_attrib_pointer(
            self.state.attrib_col as GLuint,
            4,
            gl::UNSIGNED_BYTE,
            gl::TRUE,
            self.state.vs,
            self.state.vc as GLsizeiptr,
        );

        gl::buffer_data(
            gl::ARRAY_BUFFER,
            self.vbuf.total() as GLsizeiptr,
            None,
            gl::STREAM_DRAW,
        );
        gl::buffer_data(
            gl::ELEMENT_ARRAY_BUFFER,
            self.ebuf.total() as GLsizeiptr,
            None,
            gl::STREAM_DRAW,
        );

        self.cmds.clear();
        self.vbuf.clear();
        self.ebuf.clear();
        self.config.set_null(self.null.clone());

        ctx.convert(&mut self.cmds, &mut self.vbuf, &mut self.ebuf, &self.config);

        let (_, vlen, _, _) = self.vbuf.info();
        let (_, elen, _, _) = self.ebuf.info();

        let vbytes = unsafe {
            std::slice::from_raw_parts::<u8>(self.vbuf.memory_const() as *const u8, vlen)
        };
        let ebytes = unsafe {
            std::slice::from_raw_parts::<u8>(self.ebuf.memory_const() as *const u8, elen)
        };
        gl::buffer_sub_data(gl::ARRAY_BUFFER, 0, vbytes);
        gl::buffer_sub_data(gl::ELEMENT_ARRAY_BUFFER, 0, ebytes);

        let mut eptr: *mut u16 = std::ptr::null_mut();
        for cmd in ctx.draw_command_iterator(&self.cmds) {
            if cmd.elem_count() < 1 {
                continue;
            }

            let count = cmd.elem_count();
            let mut id = cmd.texture().id().unwrap();
            self.clip_rect(cmd.clip_rect(), options);
            gl::bind_texture(gl::TEXTURE_2D, id as GLuint);
            gl::draw_elements(
                gl::TRIANGLES,
                count as GLsizei,
                gl::UNSIGNED_SHORT,
                eptr as GLsizeiptr,
            );
            eptr = unsafe { eptr.add(count as usize) };
        }

        gl::disable(gl::BLEND);
        gl::enable(gl::CULL_FACE);
        gl::enable(gl::DEPTH_TEST);
        gl::disable(gl::SCISSOR_TEST);
    }

    pub fn add_font_texture(&mut self, data: &[u8], width: u32, height: u32) -> Handle {
        self.state.add_font_texture(data, width, height)
    }

    pub fn bake_font_atlas(&mut self, atlas: &mut FontAtlas) -> Handle {
        let (image, w, h) = atlas.bake(FontAtlasFormat::Rgba32);
        let handle = self.add_font_texture(image, w, h);
        atlas.end(handle, Some(&mut self.null));
        handle
    }

    #[inline]
    pub fn clip_rect(&self, rect: &Rect, options: &DrawOptions) {
        let fx = options.dpi_factor.0 * options.scale_factor.0;
        let fy = options.dpi_factor.1 * options.scale_factor.1;
        let x = (rect.x * fx) as GLint;
        let y = ((options.display_size.1 as f32 - (rect.y + rect.h)) * fy) as GLint;
        let w = (rect.w * fx) as GLsizei;
        let h = (rect.h * fy) as GLsizei;
        gl::scissor(x, y, w, h);
    }

    pub fn ortho(&self, options: &DrawOptions) -> [f32; 16usize] {
        let fx = options.dpi_factor.0 * options.scale_factor.0;
        let fy = options.dpi_factor.1 * options.scale_factor.1;
        let matrix = [
            2.0f32 / options.display_size.0 as f32 * fx,
            0.0f32,
            0.0f32,
            0.0f32, // 1
            0.0f32,
            -2.0f32 / options.display_size.1 as f32 * fy,
            0.0f32,
            0.0f32, // 2
            0.0f32,
            0.0f32,
            -1.0f32,
            0.0f32, // 3
            -1.0f32,
            1.0f32,
            0.0f32,
            1.0f32, // 4
        ];
        matrix
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
