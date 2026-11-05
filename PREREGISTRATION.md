# Pre-registered derivations (written BEFORE any code was run)

Purpose: freeze the hand-derived predictions so the audit can show which ones
survived contact with the numerics. Nothing in this file may be edited after
coding starts; corrections go in AUDIT.md.

## Setting
Forward OU: dX = -X dt + sqrt(2) dW, X_0 ~ N(mu, Sigma). Per eigen-direction
(eigenvalue lam, mean component mu):
  sigma_t^2 = 1 + (lam-1) e^{-2t} =: 1 + q,   q := (lam-1) e^{-2t},   mean e^{-t} mu.
Reverse time s = T - t. Prior init Y_0 ~ N(0, 1).

Reverse SDE, Euler–Maruyama, step h, score at left endpoint t_k = T - k h:
  Y_{k+1} = Y_k + h [ b_k Y_k + 2 e^{-t_k} mu / sigma^2 ] + sqrt(2h) xi,  b = 1 - 2/sigma^2
  v_{k+1} = (1 + h b_k)^2 v_k + 2h
PF-ODE, Euler:  b~ = 1 - 1/sigma^2, c = e^{-t} mu / sigma^2, v_{k+1} = (1 + h b~_k)^2 v_k.

## Prediction P1 (identity): b^2 + db/ds = 1 for the SDE drift coefficient.

## Prediction P2 (SDE variance, T -> inf):
  v_N - lam = h * c_v(lam) + O(h^2),
  c_v(lam) = (2 lam - 2 - lam^2 ln lam) / (2 (lam - 1)),  c_v(1) = 1/2,  c_v(0+) = 1.
  Finite T: c_v = lam^2/(2(lam-1)) [F(lam-1) - F(q_T)],  F(q) = -ln(1+q) + 2/(1+q) - 2/(1+q)^2.

## Prediction P3 (SDE mean, T -> inf):  m_N - mu = -h mu lam ln(lam) / (4 (lam-1)) + O(h^2).

## Prediction P4 (ODE variance, T -> inf): v_N - lam = -(h/2)(lam ln lam + lam - 1) + O(h^2).

## Prediction P5 (ODE mean, T -> inf):  m_N - mu = -(h mu sqrt(lam)/2) A(lam),
  A(lam) = int_0^1 dx / sqrt(1 + (lam-1) x^2)
         = asinh(sqrt(lam-1))/sqrt(lam-1)  (lam>1),  arcsin(sqrt(1-lam))/sqrt(1-lam) (lam<1),  1 (lam=1).

## Prediction P6 (KL constant): KL(Law(Y_N) || p_data) = h^2 sum_i C(lam_i, mu_i) + O(h^3), with
  C = (dm/h)^2 / (2 lam) + (dv/h)^2 / (4 lam^2).
  Small-lam limit of the variance part: SDE ~ 1/(4 lam^2), ODE ~ 1/(16 lam^2)  (SDE 4x worse).

## Prediction P7 (prior/initialisation error, h -> 0, exact continuous flow):
  SDE: dm = -(lam e^{-2T} / sigma_T^2) mu,   dv = lam^2 e^{-2T} (1 - sigma_T^2) / sigma_T^4  => KL = Theta(e^{-4T})
  ODE: the flow is an affine bijection, so KL(out || data) = KL(N(0,1) || p_T) EXACTLY  => Theta(e^{-2T}) when mu != 0.
  Hence the data-processing bound KL <= KL(N(0,1)||p_T) is tight for the ODE and loose by ~e^{2T}/lam for the SDE mean.

## Prediction P8 (stability): EM-SDE needs h < ~lam (b -> -2/lam at t=0); Euler-ODE needs h < ~2 lam.

## Prediction P9 (non-Gaussian, ODE): for any smooth 1D target,
  KL = (h^2/2) int gamma(x) ( e'(x)/Phi'(x) + e(x) s_0(Phi(x)) )^2 dx + O(h^3),
  where Phi is the exact PF flow map, e solves  de/ds = a(s,Y) e - (1/2) Y''(s)  along characteristics, e(0)=0.
