/* Frontend-only canonical depth renderer.
 *
 * The server sends uint16 little-endian depth codes. This module performs
 * only the display conversion: code -> 8-bit code -> OpenCV JET LUT -> RGBA.
 * No colorized depth frame is uploaded, persisted, or returned by the API.
 */
(function () {
    const DEPTH_QMAX = 4095;
    let lut = null;
    let lutPromise = null;

    const VERTEX_SHADER = `#version 300 es
        in vec2 a_position;
        out vec2 v_uv;
        void main() {
            v_uv = a_position * 0.5 + 0.5;
            gl_Position = vec4(a_position, 0.0, 1.0);
        }
    `;
    const FRAGMENT_SHADER = `#version 300 es
        precision highp float;
        precision highp usampler2D;
        in vec2 v_uv;
        uniform usampler2D u_depth;
        uniform sampler2D u_lut;
        out vec4 out_color;
        void main() {
            // Typed-array texture rows start at the top for this presentation
            // path; flip only the texture lookup, never the stored samples.
            uint code = texture(u_depth, vec2(v_uv.x, 1.0 - v_uv.y)).r;
            uint c8 = min(code, 4095u) * 255u / 4095u;
            out_color = texelFetch(u_lut, ivec2(int(c8), 0), 0);
        }
    `;

    function compileShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    function createGpuState(canvas) {
        const gl = canvas.getContext('webgl2', {
            alpha: false, antialias: false, depth: false,
            preserveDrawingBuffer: false,
        });
        if (!gl) return null;
        const vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
        const fragment = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
        if (!vertex || !fragment) return null;
        const program = gl.createProgram();
        gl.attachShader(program, vertex);
        gl.attachShader(program, fragment);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null;

        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1, -1, 1, -1, -1, 1, 1, 1,
        ]), gl.STATIC_DRAW);
        const depthTexture = gl.createTexture();
        const lutTexture = gl.createTexture();
        const position = gl.getAttribLocation(program, 'a_position');
        const state = { gl, program, buffer, depthTexture, lutTexture,
            position, depthWidth: 0, depthHeight: 0, depthAllocated: false };

        gl.useProgram(program);
        gl.uniform1i(gl.getUniformLocation(program, 'u_depth'), 0);
        gl.uniform1i(gl.getUniformLocation(program, 'u_lut'), 1);
        gl.bindTexture(gl.TEXTURE_2D, lutTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        // OpenCV gives BGR; upload the equivalent RGBA display order.
        const lutPixels = new Uint8Array(256 * 4);
        lut.forEach((bgr, i) => {
            lutPixels[i * 4] = bgr[2];
            lutPixels[i * 4 + 1] = bgr[1];
            lutPixels[i * 4 + 2] = bgr[0];
            lutPixels[i * 4 + 3] = 255;
        });
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0,
            gl.RGBA, gl.UNSIGNED_BYTE, lutPixels);
        return state;
    }

    function renderGpu(canvas, codes, width, height) {
        let state = canvas._depthWebglState;
        if (!state) {
            state = createGpuState(canvas);
            if (!state) return false;
            canvas._depthWebglState = state;
        }
        const { gl } = state;
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        gl.viewport(0, 0, width, height);
        gl.useProgram(state.program);
        gl.bindBuffer(gl.ARRAY_BUFFER, state.buffer);
        gl.enableVertexAttribArray(state.position);
        gl.vertexAttribPointer(state.position, 2, gl.FLOAT, false, 0, 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, state.depthTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        if (state.depthWidth !== width || state.depthHeight !== height) {
            // Allocate immutable storage once. Repeated texImage2D calls can
            // force the browser to retire/recreate the texture and stall the
            // compositor. The frame path below only updates the existing
            // storage with texSubImage2D.
            if (state.depthAllocated) {
                // Immutable storage cannot be resized. This is normally only
                // reached if a source changes resolution after mount.
                gl.deleteTexture(state.depthTexture);
                state.depthTexture = gl.createTexture();
                gl.bindTexture(gl.TEXTURE_2D, state.depthTexture);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            }
            if (typeof gl.texStorage2D === 'function') {
                gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R16UI, width, height);
                state.depthAllocated = true;
            } else {
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, width, height, 0,
                    gl.RED_INTEGER, gl.UNSIGNED_SHORT, null);
                state.depthAllocated = false;
            }
            state.depthWidth = width;
            state.depthHeight = height;
        }
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height,
            gl.RED_INTEGER, gl.UNSIGNED_SHORT, codes);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, state.lutTexture);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        return true;
    }

    function buildCpuLut() {
        const values = new Uint32Array(DEPTH_QMAX + 1);
        const littleEndian = new Uint8Array(
            new Uint32Array([0x01020304]).buffer)[0] === 0x04;
        for (let code = 0; code <= DEPTH_QMAX; code++) {
            const bgr = lut[Math.floor(code * 255 / DEPTH_QMAX)];
            const r = bgr[2], g = bgr[1], b = bgr[0];
            values[code] = littleEndian
                ? (r | (g << 8) | (b << 16) | 0xff000000)
                : 0;
        }
        return { values, littleEndian };
    }

    function loadJetLut() {
        if (lut) return Promise.resolve(lut);
        if (lutPromise) return lutPromise;
        lutPromise = fetch('/api/v1/video/depth-jet-lut')
            .then(response => {
                if (!response.ok) throw new Error('JET LUT request failed');
                return response.json();
            })
            .then(data => {
                if (!data || data.name !== 'opencv_colormap_jet' ||
                    data.order !== 'bgr' || !Array.isArray(data.values) ||
                    data.values.length !== 256) {
                    throw new Error('Invalid OpenCV JET LUT');
                }
                lut = data.values;
                return lut;
            });
        return lutPromise;
    }

    function render(canvas, codes, width, height) {
        if (!canvas || !codes || !lut || codes.length !== width * height) {
            return false;
        }
        try {
            if (renderGpu(canvas, codes, width, height)) return true;
        } catch (_) {
            canvas._depthWebglState = null;
        }
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        const ctx = canvas.getContext('2d');
        if (!ctx) return false;
        let image = canvas._depthImageData;
        if (!image || image.width !== width || image.height !== height) {
            image = canvas._depthImageData = ctx.createImageData(width, height);
        }
        const cpuLut = canvas._depthCpuLut ||
            (canvas._depthCpuLut = buildCpuLut());
        if (cpuLut.littleEndian) {
            const pixels = new Uint32Array(image.data.buffer);
            for (let src = 0; src < codes.length; src++) {
                const code = Math.max(0, Math.min(DEPTH_QMAX, codes[src]));
                pixels[src] = cpuLut.values[code];
            }
        } else {
            // Extremely unusual host fallback; browsers in production are
            // little-endian, but keep the renderer correct everywhere.
            const pixels = image.data;
            for (let src = 0, dst = 0; src < codes.length; src++, dst += 4) {
                const code = Math.max(0, Math.min(DEPTH_QMAX, codes[src]));
                const bgr = lut[Math.floor(code * 255 / DEPTH_QMAX)];
                pixels[dst] = bgr[2];
                pixels[dst + 1] = bgr[1];
                pixels[dst + 2] = bgr[0];
                pixels[dst + 3] = 255;
            }
        }
        ctx.putImageData(image, 0, 0);
        return true;
    }

    window.DepthRenderer = { DEPTH_QMAX, loadJetLut, render };
})();
