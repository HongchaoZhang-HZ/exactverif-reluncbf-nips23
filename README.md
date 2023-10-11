# Exact Verification of ReLU NCBFs
Exact Verification of ReLU Neural Control Barrier Functions (NeurIPS 2023)

## Environment Setup

## Experiment Settings
**Darboux:** We consider the Darboux system~\cite{zeng2016darboux}, a nonlinear open-loop polynomial system that has been widely used as a benchmark for constructing barrier certificates. The dynamic model is given in the supplement. We obtain the trained NCBF by following the method proposed in \cite{zhao2020synthesizing}. 

**Obstacle Avoidance:** We evaluate our proposed method on a controlled system~\cite{barry2012safety}. We consider an Unmanned Aerial Vehicles (UAVs) avoiding collision with a tree trunk. We model the system as a  Dubins-style \cite{dubins1957curves} aircraft model. The system state  consists of 2-D position and aircraft yaw rate $x:=[x_1, x_2, \psi]^T$. We let $u$ denote the control input to manipulate yaw rate and the dynamics defined in the supplement. 
We train the NCBF via the method proposed in \cite{zhao2020synthesizing} with $v$ assumed to be $1$ and the control law $u$ designed as
 $u=\mu_{nom}(x)=-\sin \psi+3 \cdot \frac{x_1 \cdot \sin \psi+x_2 \cdot \cos \psi}{0.5+x_1^2+x_2^2}$. 

**Spacecraft Rendezvous:** We evaluate our approach on a spacecraft rendezvous problem from~\cite{jewison2016spacecraft}. A station-keeping controller is required to keep the "chaser" satellite within a certain relative distance to the "target" satellite. The state of the chaser is expressed relative to the target using linearized Clohessy–Wiltshire–Hill equations, with state $x=[p_x, p_y, p_z, v_x, v_y, v_z]^T$, control input $u=[u_x, u_y, u_z]^T$ and dynamics defined in the supplement. We train the NCBF as in~\cite{dawson2023safe}. 

**hi-ord $_8$:** We evaluate our approach on an eight-dimensional system that first appeared in \cite{abate2021fossil} to evaluate the scalability of proposed verification method. 
