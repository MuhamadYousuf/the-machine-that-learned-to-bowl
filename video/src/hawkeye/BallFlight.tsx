import React, {useEffect, useMemo} from 'react';
import * as THREE from 'three';
import {Delivery, flightEnd, pathUntil, positionAt, reachesStumps, shotPoint, shotTime, toScene} from '../lib/data';
import {palette} from '../lib/palette';

const BALL_RADIUS = 0.036;

function tube(points: [number, number, number][], radius: number, radial = 10): THREE.TubeGeometry | null {
  if (points.length < 2) return null;
  const vecs = points.map((p) => new THREE.Vector3(...p));
  // Remove duplicate consecutive points; they make the curve degenerate.
  const clean = vecs.filter((v, i) => i === 0 || v.distanceToSquared(vecs[i - 1]) > 1e-10);
  if (clean.length < 2) return null;
  const curve = new THREE.CatmullRomCurve3(clean, false, 'centripetal');
  return new THREE.TubeGeometry(curve, Math.max(4, clean.length * 2), radius, radial, false);
}

export type FlightStyle = {
  ballScale?: number;
  trailRadius?: number;
  trailOpacity?: number;
  showBall?: boolean;
  showShadow?: boolean;
  colorPre?: string;
  colorPost?: string;
  /** Persisted "all deliveries" view: draw the full path, dimmed, without the ball. */
  ghost?: boolean;
  showFlash?: boolean;
  showRing?: boolean;
  dotOpacity?: number;
  dotRadius?: number;
};

/**
 * One delivery at time `t` seconds after release.
 * Pre-bounce and post-bounce trail segments use different tints. LBW balls get a
 * dashed "predicted path" after the crease, Hawk-Eye style.
 */
export const BallFlight: React.FC<{d: Delivery; t: number; style?: FlightStyle}> = ({d, t, style = {}}) => {
  const {
    ballScale = 1.6,
    trailRadius = 0.022,
    trailOpacity = 0.95,
    showBall = true,
    showShadow = true,
    colorPre = palette.trailPre,
    colorPost = palette.trailPost,
    ghost = false,
    showFlash = true,
    showRing = true,
    dotOpacity = 0.95,
    dotRadius = 0.06,
  } = style;
  const end = flightEnd(d);
  const realEnd = d.result === 'lbw' ? Math.min(d.tc ?? end, end) : end;
  const tt = Math.max(0, Math.min(t, end));
  const tb = d.tb ?? Infinity;
  const tq = Math.round(tt * 2000) / 2000;

  const pre = useMemo(() => tube(pathUntil(d, Math.min(tq, tb, realEnd)), trailRadius), [d, tq, tb, realEnd, trailRadius]);
  const post = useMemo(
    () => (tq > tb ? tube(pathUntil(d, Math.min(tq, realEnd), tb), trailRadius) : null),
    [d, tq, tb, realEnd, trailRadius],
  );
  const predicted = useMemo(
    () => (d.result === 'lbw' && tq > realEnd ? tube(pathUntil(d, tq, realEnd), trailRadius * 1.15, 10) : null),
    [d, tq, realEnd, trailRadius],
  );

  useEffect(() => () => pre?.dispose(), [pre]);
  useEffect(() => () => post?.dispose(), [post]);
  useEffect(() => () => predicted?.dispose(), [predicted]);

  const pos = positionAt(d, tt);
  const bounced = tt >= tb && d.bounce;
  const sinceBounce = tt - tb;
  const ringT = bounced ? Math.min(1, sinceBounce / 0.35) : 0;
  const bouncePos = d.bounce ? toScene(d.bounce[0], d.bounce[1], 0) : null;
  const atEnd = t >= end;
  const sinceEnd = t - end;
  const flash = showFlash && atEnd && !reachesStumps(d) ? Math.max(0, 1 - sinceEnd / 0.25) : 0;
  // The struck ball: runs off the bat travel out to where the simulator says they went.
  const shotLen = ghost ? 0 : shotTime(d);
  const shotU = shotLen > 0 && atEnd ? Math.min(1, sinceEnd / shotLen) : 0;
  const shotQ = Math.round(shotU * 120) / 120;
  const contact = positionAt(d, end);
  const shotGeo = useMemo(() => {
    if (shotQ <= 0) return null;
    const n = Math.max(2, Math.ceil(shotQ * 40));
    return tube(Array.from({length: n + 1}, (_, i) => shotPoint(d, contact, (shotQ * i) / n)), trailRadius * 1.2);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d, shotQ, trailRadius]);
  useEffect(() => () => shotGeo?.dispose(), [shotGeo]);
  const shotPos = shotU > 0 ? shotPoint(d, contact, shotU) : null;
  const shotColour = d.runs >= 6 ? palette.pink : palette.orange;

  return (
    <group>
      {pre ? (
        <mesh geometry={pre}>
          <meshBasicMaterial color={colorPre} transparent opacity={ghost ? trailOpacity * 0.55 : trailOpacity} toneMapped={false} />
        </mesh>
      ) : null}
      {post ? (
        <mesh geometry={post}>
          <meshBasicMaterial color={colorPost} transparent opacity={ghost ? trailOpacity * 0.55 : trailOpacity} toneMapped={false} />
        </mesh>
      ) : null}
      {predicted ? (
        <mesh geometry={predicted}>
          <meshBasicMaterial color={palette.wicket} transparent opacity={0.6} toneMapped={false} />
        </mesh>
      ) : null}
      {bouncePos && bounced ? (
        <group position={[bouncePos[0], 0.006, bouncePos[2]]} rotation={[-Math.PI / 2, 0, 0]}>
          <mesh>
            <circleGeometry args={[dotRadius, 32]} />
            <meshBasicMaterial color={colorPost} transparent opacity={ghost ? 0.7 * dotOpacity : dotOpacity} toneMapped={false} />
          </mesh>
          {showRing && !ghost && ringT < 1 ? (
            <mesh>
              <ringGeometry args={[0.06 + ringT * 0.35, 0.08 + ringT * 0.37, 48]} />
              <meshBasicMaterial color={colorPost} transparent opacity={(1 - ringT) * 0.9} toneMapped={false} />
            </mesh>
          ) : null}
        </group>
      ) : null}
      {showShadow && !ghost ? (
        <mesh position={[pos[0], 0.005, pos[2]]} rotation={[-Math.PI / 2, 0, 0]}>
          <circleGeometry args={[BALL_RADIUS * ballScale * (1 + pos[1] * 0.25), 24]} />
          <meshBasicMaterial color="#000000" transparent opacity={Math.max(0.12, 0.6 - pos[1] * 0.18)} />
        </mesh>
      ) : null}
      {shotGeo ? (
        <mesh geometry={shotGeo}>
          <meshBasicMaterial color={shotColour} transparent opacity={0.9} toneMapped={false} />
        </mesh>
      ) : null}
      {shotPos && shotU < 1 ? (
        <mesh position={shotPos}>
          <sphereGeometry args={[BALL_RADIUS * ballScale * 1.2, 24, 16]} />
          <meshStandardMaterial color={palette.ball} emissive={palette.ball} emissiveIntensity={0.8} roughness={0.35} />
        </mesh>
      ) : null}
      {showBall && !ghost && !shotPos && !(atEnd && sinceEnd > 0.6) ? (
        <mesh position={pos}>
          <sphereGeometry args={[BALL_RADIUS * ballScale, 24, 16]} />
          <meshStandardMaterial color={palette.ball} emissive={palette.ball} emissiveIntensity={0.6} roughness={0.35} />
        </mesh>
      ) : null}
      {flash > 0 ? (
        <mesh position={pos}>
          <sphereGeometry args={[0.05 + (1 - flash) * 0.13, 24, 16]} />
          <meshBasicMaterial color="#ffffff" transparent opacity={flash * 0.55} toneMapped={false} />
        </mesh>
      ) : null}
    </group>
  );
};
