// WebGL post pass over the 2D map canvas: radial chromatic aberration (edges only) plus a tilt-shift /
// edge-blur depth-of-field. Bloom is done in 2D (mapDraw.bloom) before this pass.
// createFx returns null when WebGL is unavailable or ?fx=0, and the caller keeps showing the 2D canvas.
export interface Fx { render(src: HTMLCanvasElement): void; dispose(): void }

const VS = 'attribute vec2 p;varying vec2 vUv;void main(){vUv=p*.5+.5;gl_Position=vec4(p,0.,1.);}';
const FS = `precision mediump float;
varying vec2 vUv; uniform sampler2D uTex; uniform vec2 uPx;
void main(){
  vec2 c = vUv - .5;
  float r2 = dot(c, c) * 4.0;                                   // 0 at centre, 2 at corners
  vec2 ca = c * 0.012 * smoothstep(0.5, 2.0, r2);               // chromatic offset, edges only
  float b = max(smoothstep(0.22, 0.5, abs(c.y)), smoothstep(0.6, 1.5, r2)); // tilt-shift band + vignette blur
  float rad = b * 5.0;
  vec4 acc = vec4(0.0);
  for (int i = 0; i < 12; i++) {
    float fi = float(i);
    float a = fi * 2.39996;
    vec2 o = vec2(cos(a), sin(a)) * sqrt((fi + 0.5) / 12.0) * rad * uPx;
    vec4 g = texture2D(uTex, vUv + o);
    float r = texture2D(uTex, vUv + o + ca).r;
    float bl = texture2D(uTex, vUv + o - ca).b;
    acc += vec4(r, g.g, bl, g.a);
  }
  acc /= 12.0;
  acc.a = max(acc.a, max(acc.r, max(acc.g, acc.b)));            // keep premultiplied output valid after channel shifts
  gl_FragColor = acc;
}`;

// onLost is called if the GPU drops the WebGL context later; the caller must then show the plain 2D canvas again.
export function createFx(cv: HTMLCanvasElement, onLost?: () => void): Fx | null {
  try {
    if (new URLSearchParams(location.search).get('fx') === '0') return null;
    const gl = cv.getContext('webgl', { alpha: true, premultipliedAlpha: true, antialias: false });
    if (!gl) return null;
    const sh = (t: number, s: string) => { const o = gl.createShader(t)!; gl.shaderSource(o, s); gl.compileShader(o); if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(o) || 'shader'); return o; };
    const pr = gl.createProgram()!;
    gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(pr);
    if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error('link failed');
    gl.useProgram(pr);
    const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);  // one big triangle
    const loc = gl.getAttribLocation(pr, 'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    const uPx = gl.getUniformLocation(pr, 'uPx');
    cv.addEventListener('webglcontextlost', e => { e.preventDefault(); onLost?.(); }, { once: true });
    // Self-test: push a solid colour through the whole pass and read it back. Some GPUs/drivers compile the shader
    // without error but draw nothing, which would hide the map completely; in that case do not use the pass.
    {
      const probe = document.createElement('canvas'); probe.width = probe.height = 16;
      const pc = probe.getContext('2d')!; pc.fillStyle = '#ffaa33'; pc.fillRect(0, 0, 16, 16);
      cv.width = cv.height = 16; gl.viewport(0, 0, 16, 16);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, probe);
      gl.uniform2f(uPx, 1 / 16, 1 / 16); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.drawArrays(gl.TRIANGLES, 0, 3);
      const px = new Uint8Array(4); gl.readPixels(8, 8, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      if (px[0] < 120 || px[3] < 120) throw new Error('self-test: the pass produced a blank image');
    }
    return {
      render(src) {
        if (!src.width || !src.height) return;
        if (gl.isContextLost()) { onLost?.(); return; }
        if (cv.width !== src.width || cv.height !== src.height) { cv.width = src.width; cv.height = src.height; }
        gl.viewport(0, 0, cv.width, cv.height);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
        gl.uniform2f(uPx, 1 / cv.width, 1 / cv.height);
        gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      },
      // release the GPU context explicitly: browsers allow only about 16, and "Replay intro" would otherwise leak one each time
      dispose() { gl.deleteTexture(tex); gl.deleteBuffer(buf); gl.deleteProgram(pr); gl.getExtension('WEBGL_lose_context')?.loseContext(); },
    };
  } catch (e) { console.warn('fx disabled:', (e as Error).message); return null; }
}
