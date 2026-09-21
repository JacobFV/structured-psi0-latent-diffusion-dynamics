// Pure lookups joining scene bodies, robot summaries and public observations (no execution logic).
import type { RobotSummary, Scene, Snapshot } from "../transport/responses";

export interface JointRef { robot: RobotSummary; joint: RobotSummary["joints"][number]; obsIndex: number }

export function jointByName(snap: Snapshot, name: string): JointRef | null {
  for (const robot of snap.robots) {
    const joint = robot.joints.find((j) => j.name === name);
    if (joint) return { robot, joint, obsIndex: snap.observation.joints.addresses.indexOf(`${robot.index}:${joint.address}`) };
  }
  return null;
}

export function jointByAddress(snap: Snapshot, address: string): JointRef | null {
  for (const robot of snap.robots) {
    const joint = robot.joints.find((j) => j.address === address);
    if (joint) return { robot, joint, obsIndex: snap.observation.joints.addresses.indexOf(`${robot.index}:${joint.address}`) };
  }
  return null;
}

export const linkOf = (jointAddress: string) => jointAddress.split(":")[0];

export function assembliesOfLink(snap: Snapshot, link: string) {
  return snap.robots.flatMap((r) => r.assemblies.filter((a) => a.members.includes(link)).map((a) => ({ robot: r, assembly: a })));
}

export interface BodyInfo {
  id: number; name: string;
  joints: JointRef[];
  links: string[];
  assemblies: { robot: RobotSummary; assembly: RobotSummary["assemblies"][number] }[];
  object: { sim_body: string; descriptor: string; kind: string } | null;
}

export function bodyInfo(scene: Scene, snap: Snapshot, name: string): BodyInfo | null {
  const b = scene.bodies.find((x) => x.name === name);
  if (!b) return null;
  const joints = (b.joints ?? []).map((j) => jointByName(snap, j)).filter((x): x is JointRef => !!x);
  const links = [...new Set(joints.map((j) => linkOf(j.joint.address)))];
  const assemblies = links.flatMap((l) => assembliesOfLink(snap, l));
  const uniq = assemblies.filter((a, i) => assemblies.findIndex((x) => x.assembly.id === a.assembly.id && x.robot.index === a.robot.index) === i);
  const object = scene.objects?.find((o) => o.sim_body === name) ?? null;
  return { id: b.id, name: b.name, joints, links, assemblies: uniq, object };
}

/** Body that carries a given joint (via scene joint names). */
export function bodyOfJoint(scene: Scene, jointName: string): string | null {
  return scene.bodies.find((b) => (b.joints ?? []).includes(jointName))?.name ?? null;
}

export function observedObject(snap: Snapshot, descriptor: string) {
  return snap.observation.objects.find((o) => o.descriptor === descriptor) ?? null;
}
