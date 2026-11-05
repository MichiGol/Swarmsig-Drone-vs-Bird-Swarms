# Do they move together? Neighbour speed coupling as a cue to tell drone swarms from bird flocks, and a weather-robust detection system model

*Independent hobby research project · October 2026 · not peer-reviewed*

## Abstract

Small drones near airports are hard to tell apart from birds: both are small, slow radar targets that often fly in groups. We ask whether the way individuals in a group coordinate their speed reveals who is flying. As a cue we use the correlation of speed fluctuations between nearest neighbours, $T_{nn}$, after the collective motion of the group has been removed. In a Monte Carlo study with agent-based models of birds (two models, calibrated) and drones (three published swarm controllers plus formation flight), swarm-controlled drones are clearly more coupled than our bird models (AUC′ 0.85–0.99 at $N = 50$ under ideal observation). Applied with an identical pipeline to four public trajectory data sets, real research drones (0.11) are more coupled than homing pigeons (≈ 0), but less than chimney swifts (0.20) and jackdaws (0.29). On the way we found that the standard removal of rigid collective modes creates spurious coupling in non-rigid turning flight; speed-based polynomial removal fixes this. Drones that only hold a formation are uncoupled in calm air and appear coupled only in wind. A second part models a whole detection system with radar, rotor micro-Doppler, cameras and thermal imaging, driven by six years of real airport weather and real bird-migration densities. Weather costs an X-band radar little (≥ 80 % of its clear-air range during 97 % of the time); the camera's main limit is night. Without discrimination a drone radar would see about 150 unclassifiable small birds at any moment, about 270 on migration nights; with micro-Doppler this load vanishes whenever the radar does not see birds farther than it can recognise rotors.

## 1 Introduction and research question

Drone sightings have closed runways at major airports, while birds are the most common source of false alarms for drone-detection radars. Single-target classification by radar micro-Doppler or by images is well studied. Much less is known about the *group* level: drones increasingly fly in coordinated swarms, and many birds fly in flocks.

The project started from a simple thought about Monte Carlo simulation: if one can simulate drone swarms, one can ask whether an airport radar, for example an FMCW radar, could tell a group of drones from a bird flock during the murmuration season. The idea was to look at how strongly the individuals of a group are coupled to each other, and additionally at their heat signatures. This leads to two research questions:

- **RQ1 (collective cue).** Does the nearest-neighbour coupling of speed fluctuations separate drone groups from bird flocks, and does the difference survive wind, measurement noise, realistic drone behaviour and real data?
- **RQ2 (system).** How would a detection system of radar and cameras perform in real weather, how far do the different discrimination cues reach, and how many birds would it be unable to classify?

All main calculations were pre-registered: hypotheses, statistics and decision criteria were written down and committed before each run. Deviations are documented.

## 2 Methods

### 2.1 Agent-based swarm models

Each individual $i = 1,\dots,N$ is a point particle with position $\mathbf x_i$ and own velocity $\mathbf v_i$ (relative to the air for birds, the controlled velocity for drones):

$$
\mathrm d\mathbf v_i = \mathbf a_i\,\mathrm dt + \sigma\,\mathrm d\mathbf W_i,\qquad
\mathrm d\mathbf x_i = \mathbf u_i\,\mathrm dt,\qquad
\mathbf u_i = \mathbf v_i + \mathbf w(\mathbf x_i,t) - c_i\,\hat{\mathbf w}_i ,
$$

where $\mathbf u_i$ is the velocity over ground, $\mathbf w$ the wind field, $\hat{\mathbf w}_i$ a low-pass filtered wind estimate and $c_i$ the share of slow wind that is compensated ($c_D = 1$ for GPS-controlled drones, a prior for birds). The acceleration is a sum of modules, $\mathbf a_i = \mathbf a_i^{\text{align}} + \mathbf a_i^{\text{speed}} + \mathbf a_i^{\text{coh}} + \mathbf a^{\text{nav}}$, where the navigation term is identical for all individuals. Integration uses the Euler–Maruyama scheme with a stability-based time step; 30 s of burn-in are followed by 10 s of observation at 10 frames per second.

**Birds.** B1 (large flocks, starling-like) aligns with the $n_c \in \{6,7,8\}$ nearest neighbours, $\mathbf a_i^{\text{align}} = J(\langle\mathbf u\rangle_{\mathcal N_i} - \mathbf u_i)$, and regulates speed either stiffly or marginally,

$$
\mathbf a_i^{\text{speed}} = -\kappa_s\,g(s_i - s_0)\,\hat{\mathbf v}_i,\qquad g(u) = u \;\;\text{or}\;\; g(u) = u^3/u_*^2 .
$$

B3 (small flocks such as pigeons) uses hierarchical, delayed alignment with rank weights $\omega_{ij} = e^{\eta_h(h_j - h_i)}$ and delays $\tau_{ij} = \tau_0 (h_j-h_i)_+$. Both models were calibrated to polarisation, nearest-neighbour distance and stability (accepted parameter sets: 21 for B1, 45 for B3).

**Drones.** Three published controllers compute a commanded velocity from noisy, delayed neighbour data: D1 Reynolds boids, D2 Olfati-Saber flocking with a virtual leader, and D3 Vásárhelyi et al. (2018), the algorithm flown by 30 real drones. D3 combines self-propulsion, linear short-range repulsion and a velocity-alignment "friction" limited by a braking curve. A formation model DF has no neighbour term at all: each drone follows its own offset path, $\mathbf v_i^{\text{cmd}} = \mathbf u_r + k_{\text{form}}(\mathbf x_r + \mathbf o_i - \hat{\mathbf x}_i)$. A realism layer adds control rate, latency, GPS errors (Ornstein–Uhlenbeck), packet loss, first-order inertia and saturation.

**Wind and measurement.** Mean wind follows a power law in height with $U_{10}$ from autumn data of four German airports; turbulence follows the von Kármán spectrum with MIL-F-8785C intensities, synthesised as a sum of random Fourier modes and advected by the mean wind. The reference sensor is ideal tracking with Gaussian position noise $\sigma_{\text{pos}}$; velocities are central differences over $\pm q$ frames.

### 2.2 The coupling statistic

Collective modes must be removed first, because a rigidly turning or "breathing" group has velocities that vary smoothly with position and look like perfect correlation. With $\mathbf r_i = \mathbf x_i - \bar{\mathbf x}$ we compare:

- **A7**: least-squares fit of translation, rigid rotation and isotropic dilatation, $\mathbf u_i \approx \mathbf U + \boldsymbol\Omega\times\mathbf r_i + D\,\mathbf r_i$, and $\delta s_i = \hat{\mathbf n}\cdot\delta\mathbf u_i$ along the mean flight direction;
- **M**: mean removal only, $\delta s_i = |\mathbf u_i| - \bar s$;
- **L / Q** (robust, introduced in this project): the speeds $s_i = |\mathbf u_i|$ are fitted with a linear (L) or quadratic (Q) polynomial in $\mathbf r_i$, and the residual is $\delta s_i$. Q needs $N \gtrsim 30$; we report Q for $N \ge 30$, L for $N = 20$, and always both.

The statistic is the bias-corrected nearest-neighbour correlation, averaged over frames:

$$
c_{nn} = \frac{\tfrac1N\sum_i \delta s_i\,\delta s_{nn(i)}}{\tfrac1N\sum_i \delta s_i^2},\qquad
T_{nn} = \overline{c_{nn}} - \mathbb E_0\!\left[c_{nn}\right].
$$

Because the residuals sum to zero, independent fluctuations are slightly anticorrelated, $\operatorname{Corr}(\delta_i,\delta_j) = -1/(N-1)$ under mean removal and more negative for richer fits. The bias $\mathbb E_0[c_{nn}]$ is therefore estimated with a null model: a table over $N$ from independent Ornstein–Uhlenbeck velocities in simulations, and, for real data, independent isotropic fluctuations on exactly the measured positions of each frame (20 repetitions).

### 2.3 Monte Carlo design and separability

A design point fixes shared conditions (group size, flight mode, course, speed, wind, sensor noise) drawn from priors. Each class is then simulated with its own independent random numbers, derived from a seed hierarchy. Sweeps and control experiments reuse the class seeds (common random numbers), so differences are estimated precisely. Separability is

$$
\widehat{\mathrm{AUC}'} = \frac{1}{|\mathcal K|}\sum_{(a,b)\in\mathcal K}\Big[\mathbf 1\{T^{D}_a > T^{B}_b\} + \tfrac12\,\mathbf 1\{T^{D}_a = T^{B}_b\}\Big],\qquad \mathcal K = \{(a,b): a \neq b\},
$$

the probability that a random drone group is more coupled than a random bird flock from a different design point. 95 % intervals come from a bootstrap over design points (2000 resamples); for real data, flights or flocks are the clusters.

### 2.4 Real trajectory data

Four public data sets were analysed with the identical pipeline: homing pigeon flocks with GPS (Papadopoulou et al. 2022, data of Sankey et al. 2021; 7–34 birds), jackdaw flocks from stereo cameras (Ling et al. 2019; one duplicated flock excluded), chimney swifts circling before roosting (Evangelista et al. 2017; about 300 birds per frame) and a 30-drone outdoor swarm (Vásárhelyi et al. 2018). Each frame is restricted to the main cluster (single linkage at five times the median nearest-neighbour distance).

### 2.5 System model

**Radar.** The signal-to-noise ratio in energy form with coherent integration time $T_{\text{int}}$ and two-way rain attenuation is

$$
\mathrm{SNR}(R) = \frac{P\,T_{\text{int}}\,G^2\lambda^2\sigma}{(4\pi)^3R^4\,kT_0\,F\,L}\,10^{-2\gamma R/10},\qquad G = \frac{4\pi A}{\lambda^2}.
$$

For a Swerling-1 target with a single look, $P_d = P_{fa}^{1/(1+\mathrm{SNR})}$, so $P_d = 0.9$ and $P_{fa} = 10^{-6}$ require $\mathrm{SNR} \approx 130$ (21.1 dB). Rain attenuation follows ITU-R P.838-3, $\gamma = k R_r^{\alpha}$. Rain echo uses the Marshall–Palmer reflectivity $Z = 200 R_r^{1.6}$, the volume reflectivity $\eta = \pi^5|K|^2 Z\,10^{-18}/\lambda^4$, a Gaussian resolution volume $V = \frac{\pi}{8\ln 2}R^2\theta^2\frac{c}{2B}$ and a Doppler suppression $I_c$:

$$
\mathrm{SCR} = \frac{\sigma\,10^{I_c/10}}{\eta V},\qquad \mathrm{SCNR} = \left(\mathrm{SNR}^{-1} + \mathrm{SCR}^{-1}\right)^{-1}.
$$

The detection range $R_E$ is the largest $R$ with $\mathrm{SCNR} \ge 130$, found by bisection in $\log R$.

**Discrimination cues.** The rotor signature (micro-Doppler) is treated as a target of equivalent cross-section $\sigma_{\text{eq}} = \sigma\,10^{\Delta_b/10}\,\mathrm{SNR}_{\text{req}}/10^{\mathrm{SNR}_{mD}/10}$, limited only by noise and attenuation, because blade Doppler lies far outside the rain spectrum. Cameras are limited by the Johnson pixel criterion and by Koschmieder contrast loss,

$$
R_{\text{cam}} = \min\!\left(\frac{s\,f}{n\,p},\; \frac{V}{3.912}\ln\frac{C_0}{\varepsilon}\right),
$$

daytime only; thermal imaging uses the same form with longer-wave pixels and better transmission. A stereo pair with baseline $B$ has depth error $\delta Z = Z^2\delta d/(f_{px}B)$, so the swarm statistic, which needs $\delta Z \le 5$ cm, reaches $Z_{\max} = \sqrt{\sigma_{\max} f_{px} B/\delta d} \propto \sqrt{B}$. With the best discrimination range $R_U$ the *unclassified zone* is the share $1 - \min(R_U, R_E)/R_E$ of the detection range.

**Uncertainty.** Radar parameters, cross-sections of drones and birds, blade echo, camera and thermal optics are drawn from documented distributions (2000 draws, mostly log-uniform over the published range). Every result is computed per draw and reported as median with 10 % and 90 % quantiles.

### 2.6 Real weather and bird load

For each 10-minute interval of 2020–2025 at four northern German airports (1.10 million intervals), the German Weather Service (DWD) provides rain rate and visibility; the sun elevation determines day or night. Availability is the time share per draw in which a range is reached. Whether weaknesses coincide is tested by reshuffling visibility within station and month, which keeps all distributions but destroys their coupling.

Bird densities $\rho(z)$ come from weather-radar vertical profiles (Aloft) of three radars paired with the airports. With the column density $n = \int_0^{600\,\text{m}}\rho\,\mathrm dz$ and a uniform horizontal distribution, the number of birds that the system detects ($r \le R_V$) but cannot rule out as drones ($r > R_U$) is

$$
N = n\,\pi\,\max(R_V^2 - R_U^2, 0),\qquad
\Phi = 2\,n\,v\,\max(R_V - R_U, 0),
$$

where $\Phi$ counts tracks per hour that cross the detection circle without ever coming closer than $R_U$ (birds on straight paths with ground speed $v$; checked by a Monte Carlo unit test). Annual means weight all months equally.

## 3 Implementation

Everything is implemented in a Python package (`swarmsig`) with NumPy and Numba kernels for the agent dynamics; the fast kernels are tested against a readable NumPy reference. Configurations are YAML files, every random number derives from a single experiment seed, and every run writes a Parquet table of statistics and covariates. The test suite has 193 unit and integration tests, including analytic checks (Swerling threshold, ITU attenuation values, sun elevation, the bias $-1/(N-1)$, the bird-flux formula). The largest simulation studies took 15–40 minutes on a laptop. Real-data readers parse the archives as plain data only. An interactive project website reimplements the radar and optics model in JavaScript and contains a simplified toy model of coupled speed fluctuations for illustration.

## 4 Results

### 4.1 Simulation: drones are more coupled, the opposite of the starting hypothesis

The pre-registered hypothesis expected birds to be *more* coupled (marginal speed control). The first small-group study rejected this direction. A pre-registered confirmation run then established the reverse: under A7 at $N = 50$, AUC′ was 0.85 (D1), 0.89 (D2) and 0.99 (D3). The difference held in real autumn wind (0.88 / 0.88 / 0.99), with realistic GPS positioning (0.74–0.97) and with measurement noise up to about 5 cm if velocities are differenced over about ±0.5 s. It weakens when individual drones jitter strongly (0.54–0.78 at a mechanical noise of 1 m/s$^{3/2}$).

### 4.2 Real data and the turning artefact

| Group (real data) | Q (primary) | L | A7 |
|---|---|---|---|
| Homing pigeons (27 flights) | −0.00 [−0.02; 0.02] | 0.05 | 0.13 |
| Research drones (30 drones) | 0.11 [0.03; 0.19] | 0.20 | 0.60 |
| Chimney swifts | 0.20 [0.16; 0.27] | 0.31 | 0.49 |
| Jackdaws (5 flocks) | 0.29 [0.20; 0.35] | 0.48 | 0.76 |

Under A7 the real drones looked strongly coupled (0.60). A test suite with synthetic groups showed why: when a group turns non-rigidly, for example with heading differences of only 10° across the group, the rigid fit leaves smooth residuals that neighbours share, and A7 reports a coupling of 0.65 without any interaction. Mean removal (M) fails in the opposite case, for rigid rotation. The speed-based Q and L variants are robust to both. Under Q, real drones remain separable from real pigeons (AUC′ 0.79 [0.57; 0.92]), and the drone model D3 reproduces the real drones (Q 0.12 vs 0.10 in comparable conditions), while the bird models reproduce the pigeons. They do not reach the jackdaws: none of 80 parameter sets exceeded $T_L = 0.32$, against 0.48 measured. Two of three real bird species are more coupled than the real drone swarm.

### 4.3 Wind creates coupling between drones that do not interact

Formation drones without any neighbour term have $T \approx 0$ in calm air. In wind they appear coupled, because neighbours correct the same gusts: AUC′ against birds was 0.90 with exact positions, 0.88 with RTK positioning and 0.59 with standard GPS. For D3, about half of the coupling under Q is wind response. The statistic therefore measures interaction *and* shared response to wind, and its value depends on the weather.

### 4.4 No front-to-back signature

We tested whether birds follow the bird in front, which would give the lagged neighbour correlation a direction. No bird species showed this (asymmetry 0.02–0.05, real drones 0.07; AUC′ 0.38–0.47), not even at 60 frames per second. Bird speed coupling is simultaneous and slow: the correlation at ±1 s still reaches 70–80 % of its value at zero lag.

### 4.5 System model: who reaches how far

| Weather | Radar detection | Micro-Doppler | Camera | Thermal | Swarm statistic (stereo) |
|---|---|---|---|---|---|
| Clear day | 7.17 [3.71; 13.39] | 4.96 [2.48; 9.70] | 2.75 [1.32; 5.82] | 0.40 | 0.32 |
| Rain 4 mm/h | 1.83 | 4.80 | 2.15 | 0.40 | 0.32 |
| Heavy rain 25 mm/h | 0.43 | 3.80 | 0.93 | 0.40 | 0.32 |
| Fog (300 m) | 7.17 | 4.96 | 0.19 | 0.27 | 0.18 |
| Clear night | 7.17 | 4.96 | – | 0.40 | – |

*Median ranges in km for a small drone, X band, rain-echo suppression 10–30 dB (10 %/90 % quantiles in brackets for the clear day).*

In rain the rain echo, not the attenuation, limits detection, so Doppler suppression is the decisive design parameter (at 10 mm/h, 12 % of the clear-air range remains with 10–30 dB suppression and 76 % with 30–50 dB). An S-band radar is far more robust in rain but reaches only 3.4 km in clear air. Birds are detected at similar ranges as drones (a large bird beats the drone in 54 % of draws), so detection alone does not suffice. The rotor signature discriminates farthest in every weather; between detection and discrimination remains an unclassified zone of 25–31 % of the detection range. The stereo swarm statistic is a short-range cue (0.32 km at a 15 m baseline), growing with $\sqrt{B}$ to about 1.4 km at 300 m.

### 4.6 Six years of real airport weather

Precipitation fell in 7.5 % of the intervals, at least 4 mm/h in 0.3 %, and visibility was below 1 km in 1.75 %. The X-band radar kept at least 80 % of its clear-air range during 97.3 % [94.9; 99.5] of the time, the S band during 99.8 %. Parameter uncertainty matters far more than weather: 77 % of the draws detect the drone at 5 km on a clear day, and those that do, do so 98 % of the time. Confirmation by two different principles (radar micro-Doppler *and* an image) out to 1 km is available about 50 % of the time (winter 34 %, summer 65 %), essentially the daylight hours; out to 0.3 km it is 99 % thanks to thermal imaging. The weaknesses of the sensors rarely coincide: fog is 3.2 times more frequent at night and 21 times more frequent in light wind, when the camera is off anyway. Real weather coupling even raises double confirmation slightly (+0.3 to +0.4 percentage points) compared with independent weather elements, and this difference vanishes when the reshuffling is stratified by day and night.

### 4.7 Bird load

Checking the bird data revealed that the German weather-radar profiles in Aloft are apparently heavily filtered from about 2019 on: nocturnal densities in migration months are 10–100 times lower than in 2016–2018, and the 2023–2025 data fail the pre-registered plausibility check (more "birds" in rain than in dry weather). The main analysis therefore uses 2018 (decided and documented before any result was computed).

| Quantity (annual mean, small birds, 0–600 m) | Value |
|---|---|
| Unclassifiable birds at the same time, no discrimination | 146 [46; 460] |
| … on migration nights in spring and autumn | ≈ 270 |
| New tracks per hour, no discrimination | 782 [439; 1385] |
| Draws with zero load using micro-Doppler, camera and thermal | 83 % |
| Load in the remaining draws (at the same time / never-classified tracks per hour) | 34 [4; 146] / 89 [16; 288] |
| Draws with a non-empty ring for medium-sized birds | 55 % |

The load switches on sharply where the bird detection range exceeds the drone discrimination range: $R_V/R_U < 1$ is the design rule. Two thirds of the bird load falls at night and 81 % in spring and autumn, when the camera is often unavailable.

## 5 Discussion

**What holds.** Neighbour speed coupling separates swarm-controlled drones from loosely coupled flocks such as pigeons, in simulation and in real data, and it is robust to wind, GPS errors and moderate measurement noise. It does not separate drones from strongly coupled species such as jackdaws or swifts, where the birds are more coupled than the drones. For formation drones the separation depends on wind. The cue is therefore one component of a system, not a stand-alone classifier.

**Methodological contribution.** The common removal of rigid collective modes can turn non-rigid turning into apparent correlation, and mean removal fails for rigid rotation. Speed-based polynomial removal with an empirical, position-conditional bias correction is a simple remedy and should be checked in other collective-motion studies.

**System view.** No single sensor covers all conditions, but their weaknesses are complementary in real weather: rain hurts the radar, night and fog hurt the camera, and these rarely coincide. The most important system parameter for false alarms is not raw range but the ratio of bird detection range to rotor-recognition range.

**Novelty.** To our knowledge, based on a non-systematic literature search, the following have not been reported before: a measurement of this coupling on real drone swarms; a direct comparison of real drones and real birds with an identical method; the turning artefact with its test suite; and an estimate of the bird false-alarm load of a counter-drone radar from weather-radar migration profiles with parameter uncertainty and real weather.

**Limitations.** The real-data base is small (one drone data set, few flocks per species). The bird models do not reach the local coupling of jackdaws. The system model has no ground clutter, terrain or multipath, an optimistic micro-Doppler model, assumed optics, and point weather measurements; the presence of a cue is not yet a hit rate. Bird densities near the ground are extrapolated, local birds at airports are not covered, and the main bird analysis rests on one year. Radar measurement of the swarm statistic itself (only radial velocities, coarse cells) was not modelled.

## 6 Conclusion

Drones and birds can partly be told apart by how they move together: swarm controllers produce stronger neighbour coupling than loosely flying birds, but some bird species couple even more strongly, and formation drones become visible only in wind. A practical system should rely on radar with rotor micro-Doppler as its backbone, use cameras for independent confirmation by day and thermal imaging at night, and be designed so that it does not detect birds farther than it can recognise rotors. Honest checks along the way, three of which overturned intermediate results, were the most valuable part of the work.

## Statement on the use of AI

The idea, research question, project plan, sensor concept, choice of real jackdaws as the reference, and the idea for the system model in Part 2 are my own. I developed the research plan, including models, hypotheses, statistics, and validation steps; an AI chatbot was used as a sounding board for the radar equations. I located and downloaded the public flight data for real birds and drones and provided it for the analysis. I made the key scientific decisions, including which studies to run and the radar-for-detection/camera-for-discrimination concept. I used Claude (Anthropic) as an AI assistant to implement the plan: it worked out the detailed methods, wrote the code, ran the simulations and data analyses, and wrote the reports, the paper text, and the website. I specified the requirements, reviewed the results, and take full responsibility for the correctness and integrity of the work. No AI system is an author; all scientific claims and any errors are my responsibility.

## Data and code

Trajectory data: Zenodo mirrors of the Dryad data sets cited below. Weather: DWD Climate Data Center (open data). Bird profiles: Aloft (aloftdata.eu), source baltrad. Code, configurations, pre-registrations and detailed reports (in German) are kept in the project repository.

## References

- Ballerini, M. et al. (2008). Interaction ruling animal collective behavior depends on topological rather than metric distance. *PNAS* 105, 1232–1237.
- Cavagna, A. et al. (2010). Scale-free correlations in starling flocks. *PNAS* 107, 11865–11870.
- Dokter, A. M. et al. (2011). Bird migration flight altitudes studied by a network of operational weather radars. *J. R. Soc. Interface* 8, 30–43.
- Evangelista, D. J. et al. (2017). Three-dimensional trajectories and network analyses of group behaviour within chimney swift flocks during approaches to the roost. *Proc. R. Soc. B* 284, 20162602. Data: doi:10.5061/dryad.p68f8.
- ITU-R (2005). Recommendation P.838-3: Specific attenuation model for rain for use in prediction methods.
- Ling, H. et al. (2019). Local interactions and their group-level consequences in flocking jackdaws. *Proc. R. Soc. B* 286, 20190865. Data: doi:10.5061/dryad.kb8js06.
- Marshall, J. S. & Palmer, W. McK. (1948). The distribution of raindrops with size. *J. Meteorology* 5, 165–166.
- Nagy, M., Ákos, Z., Biro, D. & Vicsek, T. (2010). Hierarchical group dynamics in pigeon flocks. *Nature* 464, 890–893.
- Olfati-Saber, R. (2006). Flocking for multi-agent dynamic systems: algorithms and theory. *IEEE Trans. Automatic Control* 51, 401–420.
- Papadopoulou, M. et al. (2022). Emergence of splits and collective turns in pigeon flocks under predation. *R. Soc. Open Sci.* 9, 211898. Data from Sankey et al. 2021, doi:10.5061/dryad.905qfttkc.
- Reynolds, C. W. (1987). Flocks, herds and schools: a distributed behavioral model. *Computer Graphics* 21, 25–34.
- Swerling, P. (1960). Probability of detection for fluctuating targets. *IRE Trans. Information Theory* 6, 269–308.
- van Gasteren, H. et al. (2019). Aeroecology meets aviation safety: early warning systems in Europe and the Middle East prevent collisions between birds and aircraft. *Ecography* 42, 899–911.
- Vásárhelyi, G. et al. (2018). Optimized flocking of autonomous drones in confined environments. *Science Robotics* 3, eaat3536.
