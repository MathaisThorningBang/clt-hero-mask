/**
 * Fluid Reveal Effect
 * WebGL-based fluid simulation that masks between two images.
 * Based on Navier-Stokes GPU fluid simulation.
 */

class FluidReveal {
  constructor(canvas, finishSrc, skeletonSrc, options = {}) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl', {
      alpha: false,
      premultipliedAlpha: false,
    });

    if (!this.gl) {
      console.error('WebGL not supported');
      return;
    }

    this.config = {
      simResolution: options.simResolution || 128,
      dyeResolution: options.dyeResolution || 512,
      densityDissipation: options.densityDissipation || 0.97,
      velocityDissipation: options.velocityDissipation || 0.98,
      pressureIterations: options.pressureIterations || 20,
      splatRadius: options.splatRadius || 0.004,
      splatForce: options.splatForce || 6000,
      curl: options.curl || 30,
    };

    this.pointers = [{ x: 0, y: 0, dx: 0, dy: 0, moved: false, down: false }];
    this.lastTime = Date.now();

    this.init(finishSrc, skeletonSrc);
  }

  compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  createProgram(vertSrc, fragSrc) {
    const gl = this.gl;
    const vert = this.compileShader(gl.VERTEX_SHADER, vertSrc);
    const frag = this.compileShader(gl.FRAGMENT_SHADER, fragSrc);
    const program = gl.createProgram();
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
      return null;
    }

    const uniforms = {};
    const uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < uniformCount; i++) {
      const info = gl.getActiveUniform(program, i);
      uniforms[info.name] = gl.getUniformLocation(program, info.name);
    }

    return { program, uniforms };
  }

  createFBO(w, h, internalFormat, format, type, filter) {
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

    return { texture, fbo, width: w, height: h };
  }

  createDoubleFBO(w, h, internalFormat, format, type, filter) {
    let fbo1 = this.createFBO(w, h, internalFormat, format, type, filter);
    let fbo2 = this.createFBO(w, h, internalFormat, format, type, filter);
    return {
      get read() { return fbo1; },
      get write() { return fbo2; },
      swap() { [fbo1, fbo2] = [fbo2, fbo1]; },
    };
  }

  loadTexture(src) {
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Placeholder pixel while loading
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));

    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    };
    image.src = src;
    return texture;
  }

  init(finishSrc, skeletonSrc) {
    const gl = this.gl;

    // Enable float textures
    gl.getExtension('OES_texture_float');
    gl.getExtension('OES_texture_float_linear');

    const texType = gl.FLOAT;

    // Load images
    this.finishTexture = this.loadTexture(finishSrc);
    this.skeletonTexture = this.loadTexture(skeletonSrc);

    // Vertex shader (shared)
    const baseVert = `
      attribute vec2 aPosition;
      varying vec2 vUv;
      varying vec2 vL;
      varying vec2 vR;
      varying vec2 vT;
      varying vec2 vB;
      uniform vec2 texelSize;
      void main() {
        vUv = aPosition * 0.5 + 0.5;
        vL = vUv - vec2(texelSize.x, 0.0);
        vR = vUv + vec2(texelSize.x, 0.0);
        vT = vUv + vec2(0.0, texelSize.y);
        vB = vUv - vec2(0.0, texelSize.y);
        gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `;

    // Splat shader
    const splatFrag = `
      precision highp float;
      varying vec2 vUv;
      uniform sampler2D uTarget;
      uniform float aspectRatio;
      uniform vec3 color;
      uniform vec2 point;
      uniform float radius;
      void main() {
        vec2 p = vUv - point;
        p.x *= aspectRatio;
        vec3 splat = exp(-dot(p, p) / radius) * color;
        vec3 base = texture2D(uTarget, vUv).rgb;
        gl_FragColor = vec4(base + splat, 1.0);
      }
    `;

    // Advection shader
    const advectionFrag = `
      precision highp float;
      varying vec2 vUv;
      uniform sampler2D uVelocity;
      uniform sampler2D uSource;
      uniform vec2 texelSize;
      uniform float dt;
      uniform float dissipation;
      void main() {
        vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
        vec3 result = dissipation * texture2D(uSource, coord).rgb;
        gl_FragColor = vec4(result, 1.0);
      }
    `;

    // Divergence shader
    const divergenceFrag = `
      precision highp float;
      varying vec2 vUv;
      varying vec2 vL;
      varying vec2 vR;
      varying vec2 vT;
      varying vec2 vB;
      uniform sampler2D uVelocity;
      void main() {
        float L = texture2D(uVelocity, vL).x;
        float R = texture2D(uVelocity, vR).x;
        float T = texture2D(uVelocity, vT).y;
        float B = texture2D(uVelocity, vB).y;
        float div = 0.5 * (R - L + T - B);
        gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
      }
    `;

    // Curl shader
    const curlFrag = `
      precision highp float;
      varying vec2 vUv;
      varying vec2 vL;
      varying vec2 vR;
      varying vec2 vT;
      varying vec2 vB;
      uniform sampler2D uVelocity;
      void main() {
        float L = texture2D(uVelocity, vL).y;
        float R = texture2D(uVelocity, vR).y;
        float T = texture2D(uVelocity, vT).x;
        float B = texture2D(uVelocity, vB).x;
        float vorticity = R - L - T + B;
        gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
      }
    `;

    // Vorticity shader
    const vorticityFrag = `
      precision highp float;
      varying vec2 vUv;
      varying vec2 vL;
      varying vec2 vR;
      varying vec2 vT;
      varying vec2 vB;
      uniform sampler2D uVelocity;
      uniform sampler2D uCurl;
      uniform float curl;
      uniform float dt;
      void main() {
        float L = texture2D(uCurl, vL).x;
        float R = texture2D(uCurl, vR).x;
        float T = texture2D(uCurl, vT).x;
        float B = texture2D(uCurl, vB).x;
        float C = texture2D(uCurl, vUv).x;
        vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
        force /= length(force) + 0.0001;
        force *= curl * C;
        force.y *= -1.0;
        vec2 vel = texture2D(uVelocity, vUv).xy;
        gl_FragColor = vec4(vel + force * dt, 0.0, 1.0);
      }
    `;

    // Pressure shader (Jacobi iteration)
    const pressureFrag = `
      precision highp float;
      varying vec2 vUv;
      varying vec2 vL;
      varying vec2 vR;
      varying vec2 vT;
      varying vec2 vB;
      uniform sampler2D uPressure;
      uniform sampler2D uDivergence;
      void main() {
        float L = texture2D(uPressure, vL).x;
        float R = texture2D(uPressure, vR).x;
        float T = texture2D(uPressure, vT).x;
        float B = texture2D(uPressure, vB).x;
        float divergence = texture2D(uDivergence, vUv).x;
        float pressure = (L + R + B + T - divergence) * 0.25;
        gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
      }
    `;

    // Gradient subtract shader
    const gradientFrag = `
      precision highp float;
      varying vec2 vUv;
      varying vec2 vL;
      varying vec2 vR;
      varying vec2 vT;
      varying vec2 vB;
      uniform sampler2D uPressure;
      uniform sampler2D uVelocity;
      void main() {
        float L = texture2D(uPressure, vL).x;
        float R = texture2D(uPressure, vR).x;
        float T = texture2D(uPressure, vT).x;
        float B = texture2D(uPressure, vB).x;
        vec2 vel = texture2D(uVelocity, vUv).xy;
        vel.xy -= vec2(R - L, T - B);
        gl_FragColor = vec4(vel, 0.0, 1.0);
      }
    `;

    // Clear shader
    const clearFrag = `
      precision highp float;
      varying vec2 vUv;
      uniform sampler2D uTexture;
      uniform float value;
      void main() {
        gl_FragColor = value * texture2D(uTexture, vUv);
      }
    `;

    // Display shader — blends two images based on density mask
    const displayFrag = `
      precision highp float;
      varying vec2 vUv;
      uniform sampler2D uFinish;
      uniform sampler2D uSkeleton;
      uniform sampler2D uDensity;
      void main() {
        vec2 imageUv = vec2(vUv.x, 1.0 - vUv.y);
        float mask = texture2D(uDensity, vUv).r;
        mask = smoothstep(0.0, 0.8, mask);
        mask = clamp(mask, 0.0, 1.0);
        vec4 finish = texture2D(uFinish, imageUv);
        vec4 skeleton = texture2D(uSkeleton, imageUv);
        gl_FragColor = mix(finish, skeleton, mask);
      }
    `;

    // Compile all programs
    this.programs = {
      splat: this.createProgram(baseVert, splatFrag),
      advection: this.createProgram(baseVert, advectionFrag),
      divergence: this.createProgram(baseVert, divergenceFrag),
      curl: this.createProgram(baseVert, curlFrag),
      vorticity: this.createProgram(baseVert, vorticityFrag),
      pressure: this.createProgram(baseVert, pressureFrag),
      gradient: this.createProgram(baseVert, gradientFrag),
      clear: this.createProgram(baseVert, clearFrag),
      display: this.createProgram(baseVert, displayFrag),
    };

    // Create fullscreen quad
    const quadVerts = new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]);
    const quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);

    // Store for rendering
    this.quadBuffer = quadBuffer;

    // Initialize framebuffers
    this.initFramebuffers(texType);

    // Setup events
    this.setupEvents();

    // Start render loop
    this.animate();
  }

  initFramebuffers(texType) {
    const gl = this.gl;
    const simW = this.config.simResolution;
    const simH = this.config.simResolution;
    const dyeW = this.config.dyeResolution;
    const dyeH = this.config.dyeResolution;

    this.velocity = this.createDoubleFBO(simW, simH, gl.RGBA, gl.RGBA, texType, gl.LINEAR);
    this.pressure = this.createDoubleFBO(simW, simH, gl.RGBA, gl.RGBA, texType, gl.LINEAR);
    this.divergenceFBO = this.createFBO(simW, simH, gl.RGBA, gl.RGBA, texType, gl.NEAREST);
    this.curlFBO = this.createFBO(simW, simH, gl.RGBA, gl.RGBA, texType, gl.NEAREST);
    this.density = this.createDoubleFBO(dyeW, dyeH, gl.RGBA, gl.RGBA, texType, gl.LINEAR);
  }

  blit(target) {
    const gl = this.gl;
    if (target) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      gl.viewport(0, 0, target.width, target.height);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
  }

  useProgram(prog) {
    const gl = this.gl;
    gl.useProgram(prog.program);
    return prog.uniforms;
  }

  setupEvents() {
    const canvas = this.canvas;

    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = 1.0 - (e.clientY - rect.top) / rect.height;
      const pointer = this.pointers[0];
      pointer.dx = (x - pointer.x) * 10.0;
      pointer.dy = (y - pointer.y) * 10.0;
      pointer.x = x;
      pointer.y = y;
      pointer.moved = true;
    });

    canvas.addEventListener('mouseenter', () => {
      this.pointers[0].down = true;
    });

    canvas.addEventListener('mouseleave', () => {
      this.pointers[0].down = false;
    });

    canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const touch = e.touches[0];
      const x = (touch.clientX - rect.left) / rect.width;
      const y = 1.0 - (touch.clientY - rect.top) / rect.height;
      const pointer = this.pointers[0];
      pointer.dx = (x - pointer.x) * 10.0;
      pointer.dy = (y - pointer.y) * 10.0;
      pointer.x = x;
      pointer.y = y;
      pointer.moved = true;
      pointer.down = true;
    }, { passive: false });

    canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const touch = e.touches[0];
      const x = (touch.clientX - rect.left) / rect.width;
      const y = 1.0 - (touch.clientY - rect.top) / rect.height;
      const pointer = this.pointers[0];
      pointer.x = x;
      pointer.y = y;
      pointer.dx = 0;
      pointer.dy = 0;
      pointer.moved = true;
      pointer.down = true;
    }, { passive: false });
  }

  splat(x, y, dx, dy) {
    const gl = this.gl;

    // Splat velocity
    let u = this.useProgram(this.programs.splat);
    gl.uniform1i(u.uTarget, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
    gl.uniform1f(u.aspectRatio, this.canvas.width / this.canvas.height);
    gl.uniform2f(u.point, x, y);
    gl.uniform3f(u.color, dx * this.config.splatForce, dy * this.config.splatForce, 0.0);
    gl.uniform1f(u.radius, this.config.splatRadius);
    this.blit(this.velocity.write);
    this.velocity.swap();

    // Splat density
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.density.read.texture);
    gl.uniform3f(u.color, 0.8, 0.0, 0.0);
    gl.uniform1f(u.radius, this.config.splatRadius * 2.0);
    this.blit(this.density.write);
    this.density.swap();
  }

  step(dt) {
    const gl = this.gl;
    const simW = this.config.simResolution;
    const simH = this.config.simResolution;
    const dyeW = this.config.dyeResolution;
    const dyeH = this.config.dyeResolution;

    // Curl
    let u = this.useProgram(this.programs.curl);
    gl.uniform2f(u.texelSize, 1.0 / simW, 1.0 / simH);
    gl.uniform1i(u.uVelocity, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
    this.blit(this.curlFBO);

    // Vorticity confinement
    u = this.useProgram(this.programs.vorticity);
    gl.uniform2f(u.texelSize, 1.0 / simW, 1.0 / simH);
    gl.uniform1i(u.uVelocity, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
    gl.uniform1i(u.uCurl, 1);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.curlFBO.texture);
    gl.uniform1f(u.curl, this.config.curl);
    gl.uniform1f(u.dt, dt);
    this.blit(this.velocity.write);
    this.velocity.swap();

    // Divergence
    u = this.useProgram(this.programs.divergence);
    gl.uniform2f(u.texelSize, 1.0 / simW, 1.0 / simH);
    gl.uniform1i(u.uVelocity, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
    this.blit(this.divergenceFBO);

    // Clear pressure
    u = this.useProgram(this.programs.clear);
    gl.uniform1i(u.uTexture, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.texture);
    gl.uniform1f(u.value, 0.8);
    this.blit(this.pressure.write);
    this.pressure.swap();

    // Pressure solve (Jacobi iterations)
    u = this.useProgram(this.programs.pressure);
    gl.uniform2f(u.texelSize, 1.0 / simW, 1.0 / simH);
    gl.uniform1i(u.uDivergence, 1);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.divergenceFBO.texture);
    for (let i = 0; i < this.config.pressureIterations; i++) {
      gl.uniform1i(u.uPressure, 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.texture);
      this.blit(this.pressure.write);
      this.pressure.swap();
    }

    // Gradient subtract
    u = this.useProgram(this.programs.gradient);
    gl.uniform2f(u.texelSize, 1.0 / simW, 1.0 / simH);
    gl.uniform1i(u.uPressure, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.texture);
    gl.uniform1i(u.uVelocity, 1);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
    this.blit(this.velocity.write);
    this.velocity.swap();

    // Advect velocity
    u = this.useProgram(this.programs.advection);
    gl.uniform2f(u.texelSize, 1.0 / simW, 1.0 / simH);
    gl.uniform1i(u.uVelocity, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
    gl.uniform1i(u.uSource, 0);
    gl.uniform1f(u.dt, dt);
    gl.uniform1f(u.dissipation, this.config.velocityDissipation);
    this.blit(this.velocity.write);
    this.velocity.swap();

    // Advect density
    u = this.useProgram(this.programs.advection);
    gl.uniform2f(u.texelSize, 1.0 / dyeW, 1.0 / dyeH);
    gl.uniform1i(u.uVelocity, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
    gl.uniform1i(u.uSource, 1);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.density.read.texture);
    gl.uniform1f(u.dissipation, this.config.densityDissipation);
    this.blit(this.density.write);
    this.density.swap();
  }

  render() {
    const gl = this.gl;
    const u = this.useProgram(this.programs.display);

    gl.uniform1i(u.uFinish, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.finishTexture);

    gl.uniform1i(u.uSkeleton, 1);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.skeletonTexture);

    gl.uniform1i(u.uDensity, 2);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.density.read.texture);

    this.blit(null);
  }

  animate() {
    const now = Date.now();
    let dt = (now - this.lastTime) / 1000;
    dt = Math.min(dt, 0.016);
    this.lastTime = now;

    // Handle pointer input
    const pointer = this.pointers[0];
    if (pointer.moved) {
      this.splat(pointer.x, pointer.y, pointer.dx, pointer.dy);
      pointer.moved = false;
    }

    this.step(dt);
    this.render();

    requestAnimationFrame(() => this.animate());
  }
}
