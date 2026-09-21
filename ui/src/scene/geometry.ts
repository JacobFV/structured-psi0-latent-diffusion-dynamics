import * as THREE from "three";
import type { SceneGeom } from "../transport/responses";

/** Build a three.js mesh for a MuJoCo geom (size semantics per MuJoCo; local pos/quat in body frame, quat = wxyz). */
export function geomMesh(g: SceneGeom): THREE.Mesh | null {
  const [s0, s1, s2] = g.size;
  let geo: THREE.BufferGeometry | null = null;
  let alongZ = false;   // three's capsule/cylinder axis is +Y; MuJoCo's is +Z
  switch (g.type) {
    case 0: geo = new THREE.PlaneGeometry(2 * (s0 || 5), 2 * (s1 || 5)); break;
    case 2: geo = new THREE.SphereGeometry(s0, 24, 16); break;
    case 3: geo = new THREE.CapsuleGeometry(s0, 2 * s1, 8, 16); alongZ = true; break;
    case 4: geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(s0, s1, s2); break;
    case 5: geo = new THREE.CylinderGeometry(s0, s0, 2 * s1, 24); alongZ = true; break;
    case 6: geo = new THREE.BoxGeometry(2 * s0, 2 * s1, 2 * s2); break;
    default: return null;   // meshes/hfields are not transmitted; skip rather than guess
  }
  if (alongZ) geo.rotateX(Math.PI / 2);
  const [r, gg, b, a] = g.rgba;
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(r, gg, b), transparent: a < 1, opacity: a, roughness: 0.75, metalness: 0.05,
    side: g.type === 0 ? THREE.DoubleSide : THREE.FrontSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
  mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  mesh.name = g.name;
  mesh.userData.geomId = g.id;
  return mesh;
}
