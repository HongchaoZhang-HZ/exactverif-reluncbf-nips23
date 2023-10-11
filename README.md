# Exact Verification of ReLU NCBFs
Exact Verification of ReLU Neural Control Barrier Functions (NeurIPS 2023)

## Environment Setup

## Experiment Settings
**Darboux:** We consider the Darboux system proposed by [[1]](zeng2016darboux), a nonlinear open-loop polynomial system that has been widely used as a benchmark for constructing barrier certificates. The dynamic model is given in the supplement. We obtain the trained NCBF by following the method proposed in[[2]](zhao2020synthesizing). 

**Obstacle Avoidance:** We evaluate our proposed method on a controlled system[[3]](barry2012safety). We consider an Unmanned Aerial Vehicles (UAVs) avoiding collision with a tree trunk. We model the system as a  Dubins-style [[4]](dubins1957curves) aircraft model. The system state  consists of 2-D position and aircraft yaw rate $x:=[x_1, x_2, \psi]^T$. We let $u$ denote the control input to manipulate yaw rate and the dynamics defined in the supplement. 
We train the NCBF via the method proposed in [[2]](zhao2020synthesizing) with $v$ assumed to be $1$ and the control law $u$ designed as
 $u=\mu_{nom}(x)=-\sin \psi+3 \cdot \frac{x_1 \cdot \sin \psi+x_2 \cdot \cos \psi}{0.5+x_1^2+x_2^2}$. 

**Spacecraft Rendezvous:** We evaluate our approach on a spacecraft rendezvous problem from [[5]](jewison2016spacecraft). A station-keeping controller is required to keep the "chaser" satellite within a certain relative distance to the "target" satellite. The state of the chaser is expressed relative to the target using linearized Clohessy–Wiltshire–Hill equations, with state $x=[p_x, p_y, p_z, v_x, v_y, v_z]^T$, control input $u=[u_x, u_y, u_z]^T$ and dynamics defined in the supplement. We train the NCBF as in [[6]](dawson2023safe). 

**hi-ord $_8$:** We evaluate our approach on an eight-dimensional system that first appeared in [[7]](abate2021fossil) to evaluate the scalability of proposed verification method. 

-----
## Reference
<p id="zeng2016darboux">[1]. Zeng, Xia, et al. "Darboux-type barrier certificates for safety verification of nonlinear hybrid systems." Proceedings of the 13th International Conference on Embedded Software. 2016.</p>
<p id="barry2012safety">[2]. Barry, Andrew J., Anirudha Majumdar, and Russ Tedrake. "Safety verification of reactive controllers for UAV flight in cluttered environments using barrier certificates." 2012 IEEE International Conference on Robotics and Automation. IEEE, 2012.</p>
<p id="zhao2020synthesizing">[3]. Zhao, Hengjun, et al. "Synthesizing barrier certificates using neural networks." Proceedings of the 23rd international conference on hybrid systems: computation and control. 2020.</p>
<p id="dubins1957curves">[4]. Dubins, Lester E. "On curves of minimal length with a constraint on average curvature, and with prescribed initial and terminal positions and tangents." American journal of mathematics 79.3 (1957): 497-516.</p>
<p id="jewison2016spacecraft">[5]. Jewison, Christopher, and R. Scott Erwin. "A spacecraft benchmark problem for hybrid control and estimation." 2016 IEEE 55th Conference on Decision and Control (CDC). IEEE, 2016. </p>
<p id="dawson2023safe">[6]. Dawson, Charles, Sicun Gao, and Chuchu Fan. "Safe control with learned certificates: A survey of neural lyapunov, barrier, and contraction methods for robotics and control." IEEE Transactions on Robotics (2023). </p>
<p id="abate2021fossil">[7]. Abate, Alessandro, et al. "Fossil: A tool for the verification of hybrid systems." International Conference on Computer Aided Verification. Springer, Cham, 2021.</p>
-----