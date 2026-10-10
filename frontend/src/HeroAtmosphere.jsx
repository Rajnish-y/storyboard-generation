import { useEffect, useRef } from 'react';
import * as THREE from 'three';

/** Lightweight decorative particle/orbit scene for the hero. */
export default function HeroAtmosphere() {
  const hostRef = useRef(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power' });
    } catch {
      return undefined;
    }

    const width = () => Math.max(host.clientWidth, 1);
    const height = () => Math.max(host.clientHeight, 1);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, width() / height(), 0.1, 100);
    camera.position.z = 9;

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.6));
    renderer.setSize(width(), height());
    renderer.setClearColor(0x000000, 0);
    renderer.domElement.setAttribute('aria-hidden', 'true');
    renderer.domElement.className = 'hero-three-canvas';
    host.appendChild(renderer.domElement);

    const starCount = 210;
    const positions = new Float32Array(starCount * 3);
    const pointSizes = new Float32Array(starCount);
    for (let i = 0; i < starCount; i += 1) {
      positions[i * 3] = (Math.random() - 0.5) * 16;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 7;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 6 - 1;
      pointSizes[i] = 0.4 + Math.random() * 1.4;
    }
    const starGeometry = new THREE.BufferGeometry();
    starGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    starGeometry.setAttribute('aSize', new THREE.BufferAttribute(pointSizes, 1));
    const stars = new THREE.Points(
      starGeometry,
      new THREE.PointsMaterial({ color: 0x9bc5ff, size: 0.035, transparent: true, opacity: 0.64, depthWrite: false }),
    );
    scene.add(stars);

    const orbitalGroup = new THREE.Group();
    orbitalGroup.position.set(2.7, 0.15, -2.3);
    scene.add(orbitalGroup);
    const ringSpecs = [
      { radius: 2.4, color: 0x5d9fff, opacity: 0.29, rx: 0.88, ry: 0.15 },
      { radius: 2.95, color: 0x8a72ff, opacity: 0.18, rx: 1.18, ry: 0.44 },
      { radius: 1.85, color: 0x57d6e8, opacity: 0.16, rx: 0.34, ry: 0.9 },
    ];
    const rings = ringSpecs.map((spec) => {
      const mesh = new THREE.Mesh(
        new THREE.TorusGeometry(spec.radius, 0.009, 4, 160),
        new THREE.MeshBasicMaterial({ color: spec.color, transparent: true, opacity: spec.opacity }),
      );
      mesh.rotation.set(spec.rx, spec.ry, 0.08);
      orbitalGroup.add(mesh);
      return mesh;
    });

    const core = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.82, 1),
      new THREE.MeshBasicMaterial({ color: 0x559cff, wireframe: true, transparent: true, opacity: 0.14 }),
    );
    core.scale.set(1.25, 0.88, 0.72);
    orbitalGroup.add(core);

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let raf = 0;
    let pointerX = 0;
    let pointerY = 0;
    let smoothX = 0;
    let smoothY = 0;
    const onPointer = (event) => {
      const rect = host.getBoundingClientRect();
      pointerX = ((event.clientX - rect.left) / Math.max(rect.width, 1) - 0.5) * 0.55;
      pointerY = ((event.clientY - rect.top) / Math.max(rect.height, 1) - 0.5) * 0.28;
    };
    const onResize = () => {
      camera.aspect = width() / height();
      camera.updateProjectionMatrix();
      renderer.setSize(width(), height());
    };
    const animate = (time = 0) => {
      smoothX += (pointerX - smoothX) * 0.025;
      smoothY += (pointerY - smoothY) * 0.025;
      stars.rotation.y = reduceMotion ? 0.03 : time * 0.000018;
      stars.rotation.x = smoothY * 0.12;
      orbitalGroup.rotation.y = (reduceMotion ? 0.15 : time * 0.00009) + smoothX;
      orbitalGroup.rotation.x = 0.06 + smoothY;
      core.rotation.x = time * (reduceMotion ? 0 : 0.00012);
      core.rotation.y = time * (reduceMotion ? 0 : 0.00017);
      renderer.render(scene, camera);
      if (!reduceMotion) raf = window.requestAnimationFrame(animate);
    };

    host.addEventListener('pointermove', onPointer, { passive: true });
    window.addEventListener('resize', onResize);
    animate(0);

    return () => {
      window.cancelAnimationFrame(raf);
      host.removeEventListener('pointermove', onPointer);
      window.removeEventListener('resize', onResize);
      starGeometry.dispose();
      stars.material.dispose();
      orbitalGroup.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
      renderer.dispose();
      if (renderer.domElement.parentNode === host) host.removeChild(renderer.domElement);
    };
  }, []);

  return <div className="hero-atmosphere" ref={hostRef} aria-hidden="true" />;
}
