import numpy as np
from math import atan2, atan, tan, pi, sqrt, cos, sin

from utils.utils import transform, distance


class SplineParams:
    """Parameters for a single cubic Hermite spline candidate."""

    def __init__(self, p0, p1, m0, m1, arc_length):
        self.p0 = np.array(p0, dtype=float)   # [x0, y0]
        self.p1 = np.array(p1, dtype=float)   # [x1, y1]
        self.m0 = np.array(m0, dtype=float)   # endpoint tangent at start
        self.m1 = np.array(m1, dtype=float)   # endpoint tangent at goal
        self.len = arc_length


class SplinePath:
    """
    Cubic Hermite spline path connector for Hybrid A*.

    Acts as a drop-in replacement for DubinsPath: the same four public
    methods are implemented so HybridAstar can use it unchanged.

    How it works
    ------------
    find_tangents() builds several spline candidates that differ only in
    their tangent *scale* (analogous to the four Dubins primitives LSL,
    LSR, RSL, RSR).  Each candidate is a cubic Hermite spline whose
    endpoint tangents are aligned with the car heading at start and goal,
    guaranteeing heading continuity.

    best_tangent() walks through candidates from shortest to longest,
    samples each one into waypoints, checks every waypoint for collision,
    and returns the first valid route.

    The route format returned — list of (pos, phi, m) tuples — is
    identical to the format DubinsPath returns, so car.get_path() works
    unchanged.

    is_straight_route_safe / is_turning_route_safe are kept identical to
    DubinsPath because they guard the A* arc expansions, not the spline.
    """

    def __init__(self, car, n_waypoints=40):
        self.car = car
        self.r = self.car.l / tan(self.car.max_phi)
        self.n_waypoints = n_waypoints  # spline sample density

    # ------------------------------------------------------------------
    # Public API  (same signatures as DubinsPath)
    # ------------------------------------------------------------------

    def find_tangents(self, start_pos, end_pos):
        """
        Generate spline candidates with varying tension scales.
        Returns a list of SplineParams sorted by ascending arc length.
        """
        self.start_pos = start_pos
        self.end_pos = end_pos

        x0, y0, th0 = start_pos
        x1, y1, th1 = end_pos
        dist = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Unit heading vectors at each endpoint
        d0 = np.array([cos(th0), sin(th0)])
        d1 = np.array([cos(th1), sin(th1)])

        solutions = []
        # Four tension scales — analogous to the four Dubins primitives.
        # Smaller scale → tighter, shorter curve near endpoints.
        # Larger scale → wider swing through space.
        for scale in [dist * 0.6, dist * 1.0, dist * 1.5, dist * 0.3]:
            m0 = scale * d0
            m1 = scale * d1
            length = self._arc_length([x0, y0], [x1, y1], m0, m1)
            solutions.append(SplineParams([x0, y0], [x1, y1], m0, m1, length))

        solutions.sort(key=lambda s: s.len)
        return solutions

    def best_tangent(self, solutions):
        """
        Return the shortest valid (collision-free) spline.
        Output: (route, cost, valid)  — same as DubinsPath.best_tangent().
        """
        if not solutions:
            return None, None, False

        for s in solutions:
            route, valid = self._build_route(s)
            if valid:
                return route, s.len, True

        return None, None, False

    def is_straight_route_safe(self, t1, t2):
        """Swept-rectangle collision check for a straight segment."""
        vertex1 = self.car.get_car_bounding(t1)
        vertex2 = self.car.get_car_bounding(t2)
        vertex = [vertex2[0], vertex2[1], vertex1[2], vertex1[3]]
        return self.car.env.rectangle_safe(vertex)

    def is_turning_route_safe(self, start_pos, end_pos, d, c, r):
        """Ring-sector collision check for a circular arc."""
        if not self.car.is_pos_safe(end_pos):
            return False

        rs_inner, rs_outer = self._construct_ringsectors(start_pos, end_pos, d, c, r)

        if not self.car.env.ringsector_safe(rs_inner):
            return False
        if not self.car.env.ringsector_safe(rs_outer):
            return False

        return True

    # ------------------------------------------------------------------
    # Cubic Hermite spline mathematics
    # ------------------------------------------------------------------

    @staticmethod
    def _hermite(t, p0, p1, m0, m1):
        """Evaluate position on a cubic Hermite spline at t ∈ [0, 1]."""
        t2, t3 = t * t, t * t * t
        h00 =  2 * t3 - 3 * t2 + 1
        h10 =      t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 =      t3 -     t2
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    @staticmethod
    def _hermite_d1(t, p0, p1, m0, m1):
        """First derivative (velocity) of cubic Hermite spline."""
        t2 = t * t
        h00 =  6 * t2 - 6 * t
        h10 =  3 * t2 - 4 * t + 1
        h01 = -6 * t2 + 6 * t
        h11 =  3 * t2 - 2 * t
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    @staticmethod
    def _hermite_d2(t, p0, p1, m0, m1):
        """Second derivative (acceleration) of cubic Hermite spline."""
        h00 =  12 * t - 6
        h10 =   6 * t - 4
        h01 = -12 * t + 6
        h11 =   6 * t - 2
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    def _arc_length(self, p0, p1, m0, m1, n=100):
        """Approximate arc length by summing chord lengths along the spline."""
        p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
        m0, m1 = np.array(m0, dtype=float), np.array(m1, dtype=float)
        pts = np.array([self._hermite(t, p0, p1, m0, m1)
                        for t in np.linspace(0, 1, n)])
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def _sample_spline(self, s):
        """
        Sample the spline into (x, y, theta, phi) lists.

        theta  — heading derived from the tangent direction.
        phi    — steering angle derived from signed curvature,
                 clamped to the car's physical limits.
        """
        p0, p1, m0, m1 = s.p0, s.p1, s.m0, s.m1

        xs, ys, thetas, phis = [], [], [], []

        for t in np.linspace(0.0, 1.0, self.n_waypoints):
            pt  = self._hermite(t, p0, p1, m0, m1)
            dp  = self._hermite_d1(t, p0, p1, m0, m1)
            d2p = self._hermite_d2(t, p0, p1, m0, m1)

            # Heading from tangent vector
            theta = atan2(float(dp[1]), float(dp[0]))

            # Signed curvature:  κ = (x'y'' − y'x'') / (x'^2 + y'^2)^(3/2)
            denom = (dp[0] ** 2 + dp[1] ** 2) ** 1.5
            kappa = float((dp[0] * d2p[1] - dp[1] * d2p[0]) / denom) \
                    if abs(denom) > 1e-9 else 0.0

            # Steering angle from curvature via bicycle model:  φ = atan(κ · L)
            phi = atan(kappa * self.car.l)
            phi = max(-self.car.max_phi, min(self.car.max_phi, phi))

            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
            thetas.append(theta)
            phis.append(phi)

        return xs, ys, thetas, phis

    def _build_route(self, s):
        """
        Convert a spline candidate into a route and validate it.

        The route is a list of (pos, phi, m) tuples compatible with
        car.get_path().  Collision is checked at every sampled waypoint.
        Returns (route, valid).
        """
        xs, ys, thetas, phis = self._sample_spline(s)

        # Snap the final heading to the exact goal heading so the car
        # arrives with the correct orientation.
        thetas[-1] = self.end_pos[2]

        route = []
        for i in range(1, len(xs)):
            pos = [xs[i], ys[i], thetas[i]]
            phi = phis[i - 1]   # steering that produced segment i-1 → i

            if not self.car.is_pos_safe(pos):
                return [], False

            route.append((pos, phi, 1))

        return route, True

    # ------------------------------------------------------------------
    # Arc safety helpers  (mirror of DubinsPath.construct_ringsectors)
    # ------------------------------------------------------------------

    def _construct_ringsectors(self, start_pos, end_pos, d, c, r):
        """Reproduce DubinsPath ring-sector geometry for arc safety checks."""
        x, y, theta = start_pos

        delta_theta = end_pos[2] - theta
        if d == 1 and delta_theta < 0:
            delta_theta += 2 * pi
        elif d == -1 and delta_theta > 0:
            delta_theta -= 2 * pi

        p_inner = start_pos[:2]
        id_ = 1 if d == -1 else 2
        p_outer = transform(x, y, 1.3 * self.car.l, 0.4 * self.car.l, theta, id_)

        r_inner = r - self.car.carw / 2
        r_outer = distance(p_outer, c)

        v_inner = [p_inner[0] - c[0], p_inner[1] - c[1]]
        v_outer = [p_outer[0] - c[0], p_outer[1] - c[1]]

        if d == -1:
            end_inner   = atan2(v_inner[1], v_inner[0]) % (2 * pi)
            start_inner = (end_inner + delta_theta) % (2 * pi)
            end_outer   = atan2(v_outer[1], v_outer[0]) % (2 * pi)
            start_outer = (end_outer + delta_theta) % (2 * pi)
        else:
            start_inner = atan2(v_inner[1], v_inner[0]) % (2 * pi)
            end_inner   = (start_inner + delta_theta) % (2 * pi)
            start_outer = atan2(v_outer[1], v_outer[0]) % (2 * pi)
            end_outer   = (start_outer + delta_theta) % (2 * pi)

        rs_inner = [c[0], c[1], r_inner, r, start_inner, end_inner]
        rs_outer = [c[0], c[1], r, r_outer, start_outer, end_outer]

        return rs_inner, rs_outer
