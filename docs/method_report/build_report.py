"""Build a documented snapshot of the implemented VoI diffusion IPP method.
Run make_assets.py first with the repository's plotting environment.
"""
from pathlib import Path
import json, hashlib
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'tmp/pdfs/method_report'
OUT=ROOT/'output/pdf/voi_diffusion_ipp_method.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
fontroot=Path('/System/Library/Fonts/Supplemental')
for name,file in [('Arial','Arial.ttf'),('Arial-Bold','Arial Bold.ttf'),('Arial-Italic','Arial Italic.ttf')]:
 pdfmetrics.registerFont(TTFont(name,str(fontroot/file)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
NAVY=colors.HexColor('#183447');TEAL=colors.HexColor('#087e8b');GRAY=colors.HexColor('#526675');LIGHT=colors.HexColor('#eef4f5');INK=colors.HexColor('#253943')
W,H=595.276,841.89; M=48;CW=W-2*M
styles={
 'body':ParagraphStyle('body',fontName='Arial',fontSize=10,leading=14.1,textColor=INK,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Arial',fontSize=8.5,leading=11.5,textColor=GRAY,spaceAfter=7),
 'caption':ParagraphStyle('caption',fontName='Arial',fontSize=8.5,leading=11.4,textColor=GRAY,spaceBefore=4,spaceAfter=11),
 'h1':ParagraphStyle('h1',fontName='Arial-Bold',fontSize=23,leading=27,textColor=NAVY,spaceAfter=15),
 'h2':ParagraphStyle('h2',fontName='Arial-Bold',fontSize=12.5,leading=17,textColor=TEAL,spaceBefore=8,spaceAfter=7),
 'cover':ParagraphStyle('cover',fontName='Arial-Bold',fontSize=32,leading=36,textColor=NAVY,spaceAfter=16),
 'subtitle':ParagraphStyle('subtitle',fontName='Arial',fontSize=14,leading=20,textColor=GRAY,spaceAfter=15),
 'tag':ParagraphStyle('tag',fontName='Arial-Bold',fontSize=9,leading=12,textColor=TEAL,spaceAfter=11),
 'cell':ParagraphStyle('cell',fontName='Arial',fontSize=8.6,leading=11.5,textColor=INK),
 'headcell':ParagraphStyle('headcell',fontName='Arial-Bold',fontSize=8.6,leading=11.5,textColor=colors.white),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=7.9,leading=11.8,textColor=NAVY,spaceAfter=8,leftIndent=8),
}
story=[]; sections=[];eqno=0

def p(text,kind='body'):story.append(Paragraph(text,styles[kind]))
def h(text):p(text,'h2')
def page(num,title):
 if story:story.append(PageBreak())
 sections.append((num,title));p(f'METHODS REPORT / {num:02d}','tag');p(title,'h1')
def eq(name):
 global eqno
 eqno+=1
 im=PILImage.open(ASSET/f'eq_{name}.png');iw,ih=im.size
 width=min(CW-35,iw/5.0);height=ih*width/iw
 tab=Table([[Image(str(ASSET/f'eq_{name}.png'),width=width,height=height),Paragraph(f'({eqno})',styles['small'])]],colWidths=[CW-29,29])
 tab.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),9)]));story.append(tab)
def fig(name,caption,width=CW):
 im=PILImage.open(ASSET/(name+'.png'));iw,ih=im.size
 story.append(Image(str(ASSET/(name+'.png')),width=width,height=ih*width/iw));p(caption,'caption')
def table(headers,rows,widths=None):
 widths=widths or [CW/len(headers)]*len(headers)
 data=[[Paragraph(str(x),styles['headcell']) for x in headers]]+[[Paragraph(str(x),styles['cell']) for x in r] for r in rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#cbd8dd'))]));story.append(t);story.append(Spacer(1,11))
def note(title,text):
 t=Table([[Paragraph('<b>'+title+'</b><br/>'+text,styles['body'])]],colWidths=[CW]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),.5,colors.HexColor('#b4cfd3')),('LEFTPADDING',(0,0),(-1,-1),11),('RIGHTPADDING',(0,0),(-1,-1),11),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,10))
def code(lines):
 for line in lines:p(escape(line).replace(' ','&nbsp;'),'code')

# 01
page(1,'')
# Replace the otherwise empty page title with a designed cover.
story.pop()
p('Value-of-Information<br/>Diffusion Planning','cover')
p('Persistent monitoring under learned spatial wind<br/>Method, implementation and evidence','subtitle')
p('Implementation snapshot: 17 October 2026<br/>Repository: voi-diffusion-ipp | source HEAD 3a31937d4924','small')
fig('geometry','The current development task: two large monitoring regions, two obstacles and a directional disturbance in the middle passage. The displayed physical field is available to the simulator and retrospective diagnostics, not to the planner. Brown points are the common noisy historical survey supplied to all compared methods.',width=CW-65)
h('The central idea')
p('Choose motion for the monitoring performance it can achieve <b>after learning from the wind measurements that motion will actually generate</b>. Evaluate the immediate monitoring cost and the subsequent safety-filtered continuation together. No separate information-gain reward is added.')
note('What this report documents','Full online POAM wind inference, clarity dynamics, sampled spline RPCBF, actual model-based reverse diffusion, and the compact causal policy used by the current 10-second planner. The 120-second results are a development demonstration; substantial isolated learning benefit is not yet established.')

# 02
page(2,'Scope and reading guide')
p('This report explains the implemented research method, its motivating formulation, and the current evidence. It is grounded in the supplied research discussion and draft [R1-R2], the reference methods [R3-R8], and the source and numeric archives listed at the end. Equations describe the present implementation unless a general formulation or diagnostic extension is explicitly identified.')
table(['Part','Contents'],[
 ('3-4','Physical mission, monitoring uncertainty, clarity loss and sensing.'),('5-7','POAM inference, online updates, numerical stability and coherent wind sampling.'),('8-10','Spline RPCBF, relaxed safety QP and the causal task objective.'),('11-14','Fast posterior-feedback policy, execution algorithm and MBD denoising.'),('15-17','Runtime architecture, baselines and exact development configuration.'),('18-21','Recorded mission, whole-field clarity, wind calibration and safety findings.'),('22-24','Verification, reproduction, provenance and references.')],[65,CW-65])
h('Two uncertainties, with different purposes')
p('<b>Monitoring uncertainty</b> is the uncertainty of a changing environmental quantity of interest (QoI). Its covariance grows through Q and falls through sensing; maintaining clarity is the task. <b>Wind uncertainty</b> is uncertainty about the fixed spatial disturbance that changes motion and the safety filter. Learning wind is useful only insofar as it improves the monitoring task.')
table(['Symbol','Meaning'],[('x = (p, v), u','Planar position, velocity and bounded control acceleration.'),('q, q̄, Q, R','Cell clarity, desired clarity, QoI process noise and QoI measurement intensity.'),('<i>B</i>, μ, Σ','Wind belief, predictive mean and spatial predictive covariance.'),('T<sub>L</sub>, T<sub>F</sub>, T<sub>B</sub>','Executed first stage, anticipated future window and backup horizon.'),('S<sub>o</sub>, L<sub>R</sub>, Y','Outer world count, independent RPCBF field count and common policy vector.')],[120,CW-120])
p('The 3D scene is a visualization of these planar recordings. Aircraft appearance, buildings, terrain and wetland labels do not introduce flight dynamics, terrain avoidance or a hydrology model. The numerical model has no vertical escape maneuver. Physical units below refer to the planar simulation, not the apparent scale of the rendered landscape.','small')

# 03
page(3,'Mission and physical model')
p('The intended mission is repeated monitoring of two extended regions A and B. Rapid information loss makes staying indefinitely in one region costly. Two obstacles create a short middle passage and longer bypasses. A wind-assisted outward transfer and a wind-opposed return can have different costs. The planner must discover whether learning the passage changes its useful future options.')
eq('dynamics')
p('Position and velocity are two-dimensional. Control is acceleration with independent component bounds. The field d(p) is a fixed but unknown spatial <b>acceleration disturbance</b>, measured in m/s². The current profile uses γ = 1 s⁻¹ and u<sub>max</sub> = 1.6 m/s². RK4 integrates the same spatial field at intermediate positions throughout a rollout.')
h('Why linear drag was added')
p('The original model used γ = 0. A position-only acceleration disturbance in that model does not, by itself, make a reversed rest-to-rest path require different control: time reversal preserves acceleration. Linear velocity drag breaks that symmetry. Interpreting drag relative to air velocity gives d = γw<sub>air</sub>, but the estimator still learns d, not measured air speed. This is a declared reduced model motivated by drag-aware flight models [R7]; it does not validate helicopter or fixed-wing aerodynamics.')
eq('observation')
p('The simulated wind sensor returns noisy acceleration-disturbance measurements at <b>executed positions</b>, with σ<sub>d</sub> = 0.1 m/s². Measurements are direct simulated observations, not estimates reconstructed from vehicle acceleration. Actual wind observations arrive at 5 Hz and are assimilated at 2-second boundaries. Each planning/execution period uses a frozen belief.')
h('Current physical field and prior information')
eq('physical')
p('Here t is the unit direction from A to B and n is its perpendicular. The smooth passage jet has a 20% cross component. Its formula is confined to the simulator and audits. All current arms receive the same 42 noisy observations along two outer survey curves; the passage interior is initially unobserved. The prior is therefore not a cold-start prior.')
note('Task calibration is separate from a learning result','The selected Q and footprint were compared against a bound for remaining in one region, using a task-only shuttle diagnostic. That establishes a reason to move between A and B. It does not establish that anticipated wind learning improves the resulting planner.')

# 04
page(4,'Persistent clarity dynamics')
p('For each cell i, use an independent normalized scalar QoI variance P<sub>i</sub>. A random-walk process and a location-dependent linear observation model give the continuous covariance dynamics below. Clarity is a monotone transformation of variance; the normalization fixes the reference variance scale to one [R2, R5].')
eq('riccati');eq('clarity');eq('footprint');eq('objective')
p('Q<sub>i</sub> is the corresponding diagonal entry of the QoI process-noise matrix. Larger Q means faster uncertainty growth away from sensing. R = 0.1 is the QoI observation-noise intensity; it is distinct from the wind sensor variance σ<sub>d</sub>². The current implementation propagates covariance/clarity, not a physical environmental mean field.')
fig('decay','Analytical consequences of the implemented dynamics. Left: clarity decays even with no robot nearby. Right: the equilibrium under a stationary sensor falls with distance. These are model calculations, not fitted performance curves.',width=CW)
eq('decay')
p('With Q = 0.16, clarity drops from 0.8 to 0.6 in about 2.60 seconds without sensing. Decay is algebraic, not exponential. Execution updates all 1,353 cells using RK4 along each executed segment; accumulated cost uses trapezoidal integration of the positive-part deficit.','small')

# 05
page(5,'Wind belief: full sparse POAM')
p('Wind inference uses an independent JAX implementation of POAM [R6]. Its two output components have independent GP distributions, sharing a spatial attentive kernel, inducing locations and scalar observation noise. They can have different posterior means but use the same spatial covariance. No finite-anchor projection replaces the predictive residual in the current model.')
h('Attentive kernel')
eq('kernel')
p('The network has two tanh hidden layers of width 8 and a four-component softmax output, normalized to unit Euclidean norm. In this implementation, the same normalized feature vector w supplies both the mixture weights and the instance-similarity factor. The amplitude a and network weights are trainable. The fixed RBF length scales are 0.3, 0.8, 1.3 and 1.8 m; the initial marginal standard deviation is 0.8 m/s².')
h('Inducing distribution and full prediction')
p('Let Z contain M inducing inputs, K<sub>ZZ</sub> their prior covariance, and q(u) = N(m, S) the variational distribution. For a set of queried locations X, prediction is:')
eq('predictmean');eq('predictcov')
p('The residual K<sub>XX</sub> - K<sub>XZ</sub>K<sub>ZZ</sub><super>-1</super>K<sub>ZX</sub> is essential: uncertainty remains between and away from inducing inputs. Retaining only the last term would make wind artificially certain in regions unsupported by data. Full joint prediction is used to preserve spatial correlation; pointwise variance is a display and summary, not the complete belief.')
h('Data-supported initialization and online adaptation')
p('The current mission has M = 24 inducing inputs. Pivoted Cholesky selects initial inputs from the common historical survey, requiring at least M distinct locations. At later updates, candidates are the previous inducing inputs and newly earned observations. Kernel parameters adapt between real planning calls using a variational EM objective and persistent Adam moments. The sensor-noise variance stays fixed at its known simulated value.')
eq('elbo')
p('Kernel learning uses minibatch-scaled expected Gaussian log likelihood minus the two output KL terms. During each M-step, q(u) and Z are held fixed; Adam updates the kernel network and amplitude. The next boundary transports statistics through the changed kernel.', 'small')
p('Full POAM here retains the predictive residual and online inference machinery. Finite inducing capacity, independent outputs and the chosen kernel remain approximations; exact dense inference or calibrated uncertainty is not claimed.','small')

# 06
page(6,'POAM updates and stable statistics')
p('For a newly earned batch (X, Y), POAM maintains unweighted accumulated statistics a and B. If the kernel or inducing locations changed, let P = K<sub>old</sub><super>-1</super> K<sub>current</sub>(Z<sub>old</sub>, Z<sub>new</sub>). The old matrix is the stored prior matrix from the preceding variational update. The update transports previous information before adding the new batch:')
eq('statistics');eq('posterior')
p('These expressions omit the small numerical regularizers for readability. They show that information is accumulated rather than discarded when inducing points move. The actual solves use symmetrized matrices and jitter. With a frozen kernel and fixed inputs, the formulas reduce to accumulated sparse Gaussian regression statistics; with online adaptation, transport is the POAM approximation.')
h('Square-root repair for long missions')
p('A previous 120-second attempt failed in the comparator after approximately 75 seconds: repeated float32 statistic transport caused a nominally positive matrix to lose positive definiteness. That failed run and its observation history remain archived. The current implementation stores B through a factor L and transports the factor using QR:')
eq('root')
p('The variational solve then whitens with a Cholesky factor of K<sub>ZZ</sub> and solves a matrix of the form I + FF<super>T</super>/σ<sub>d</sub>². Covariance is formed as V<super>T</super>V. This is an algebraically equivalent square-root form of the same POAM update, not a reset, forgetting factor or replacement estimator. Algebra tests and replays of retained observations passed after the repair.')
h('Real versus hypothetical updating')
table(['Real mission boundary','Hypothetical planning boundary'],[('Uses only actually measured values and executed positions.','Uses synthetic noisy values at the simulated executed positions.'),('Relocates inducing inputs; transports statistics; updates q(u).','Same statistic/variational update; inducing relocation enabled in the current profile.'),('Runs 10 minibatch kernel-training steps with persistent Adam state.','Keeps kernel hyperparameters fixed for the whole planning call.'),('Persists the model and observation history.','Keeps candidate-specific beliefs local; never writes them into the real model.')],[CW/2,CW/2])
p('The sparse update algebra has fixed-size sufficient statistics, but this repository also retains the raw history for minibatch learning and audit. Its history storage therefore grows with mission length; it should not be described as a globally constant-memory implementation.','small')

# 07
page(7,'Coherent sampling and information flow')
p('A physical wind field is spatially coherent: querying it twice at the same point returns the same value. Outer samples are possible worlds for expected task cost. Independent inner samples are the disturbance fields against which the safety filter constructs its approximate barrier. Neither is independent per-cell or per-timestep noise.')
eq('sample')
p('For each output component, g is a prior function draw independent of the inducing draw u. The Matheron correction gives the correct sparse posterior moments when g is an exact prior draw [R8]. Numerically, the code uses 32 Sobol spectral nodes per length-scale component, then corrects the prior on a dense 29 × 23 support grid (667 points). A spectral residual remains between support points. Inference itself uses the attentive kernel, not a Fourier replacement.')
h('Three distinct roles')
table(['Object','Permitted use','Forbidden shortcut'],[('Physical truth','Advance actual simulated dynamics and produce real sensor measurements.','Supply truth to the nominal controller, GP or RPCBF.'),('Outer world sample','Advance a candidate world and synthesize its possible observations.','Select future controls by world index or access unmeasured wind.'),('Inner belief samples','Construct the backup value and robust derivative inequalities.','Reuse the outer truth as the safety sample that knows which world will occur.')],[97,197,CW-294])
p('During a candidate rollout, the sampled world stays unchanged after a hypothetical update. Only the belief-dependent controller, filter and future reference change. The same underlying random numbers are reused across candidate policies to reduce variance, while outer and inner coefficient banks remain independent. Reusing random numbers is a coupling device, not permission to share hidden state.')
h('Numerical field cache')
p('The fast evaluator tabulates prior sample paths and kernel columns on a 129 × 107 grid over [-4.8, 4.8] × [-4, 4] m. Tensor Catmull-Rom cubic interpolation is C¹ across interior cells, enabling differentiation through the RPCBF construction. Inducing corrections and means are rebuilt after belief changes; real kernel changes require a new cache. Queries outside the valid interpolation interior return NaN.')
note('Approximation boundary','Continuous prior sampling and spatial interpolation are numerical approximations. The selected policy is re-evaluated using uncached field functions and all monitoring cells. This removes table interpolation from that check, but does not remove finite samples, sparse inference, time discretization or posterior miscalibration. In the current pair, maximum cached-versus-uncached selected-path displacement was 0.202 / 0.137 m; maximum finite objective disagreement was 0.601% / 0.278%. One comparator selection failed the uncached check and invoked backup.')

# 08
page(8,'RPCBF from a backup policy')
p('The safety layer follows the policy-value construction of Robust Policy Control Barrier Functions (RPCBF) [R4]. It is a separate controller from the MBD nominal policy. A failure function h<sub>j</sub>(x) is positive in an unsafe state and nonpositive in its safe set. For a circular obstacle, the implementation uses (r + margin)² - ||p - c||²; workspace walls have affine failure functions.')
eq('backup')
p('The backup damps velocity and compensates the belief mean. The current gain k<sub>b</sub> is 2. For each inner wind field, integrate the backup from the queried state over T<sub>B</sub> = 1.2 s with 8 steps of 0.15 s. Include the initial constraint value as well as all propagated samples.')
eq('cbf')
p('The tilde denotes a full not-a-knot cubic spline through the sampled constraint values along a backup trajectory. On every interval, maximize the cubic by checking both endpoints and all real stationary points inside the interval. Then take the maximum over intervals and inner fields. This is not merely the maximum of discrete samples, a first-order distance barrier, or a soft obstacle penalty.')
h('Differentiate the sampled policy value')
p('JAX differentiates the complete approximate value with respect to the initial state, including the backup rollout, spatial field queries and spline coefficients. At a unique active extremum this follows the local active branch. Ties and changes in the active field or spline interval are nonsmooth; numerical gradient tests are useful but do not turn the approximation into a globally smooth certificate.')
eq('constraint')
p('The derivative condition enforces dB<sub>j</sub>/dt ≤ -αB<sub>j</sub>. Its disturbance term is maximized over <b>all inner instantaneous sampled winds</b>, not just the field that attained the largest backup value. Position and velocity gradients are separated because input acts on acceleration. The linear-drag term is included consistently. Current α = 1.')
h('Interpretation and limits')
p('As observations change the posterior, both the backup trajectory ensemble and the derivative constraints can change. This provides one causal route by which learning can improve task performance. But a finite backup horizon, three inner fields, approximate interpolation and a Gaussian model do not establish uniform disturbance coverage or controlled invariance. This implementation makes empirical safety claims only; it does not inherit a formal guarantee from the reference paper.')

# 09
page(9,'Soft safety and executable control')
p('At each control tick, the filter projects the proposed nominal input onto the current barrier half-spaces and the actuator box:')
eq('qp')
p('Because there are two input dimensions, the implementation enumerates candidate optima with zero, one or two active inequalities and selects the closest feasible input. The box constraints are included explicitly. This is a genuine constrained QP solution specialized to 2D; it is not component clipping used as a substitute for the barrier constraints.')
h('When the original inequalities are infeasible')
eq('slack')
p('The actuator box remains hard. A common barrier slack is found by 18 bisection iterations. The fallback then projects the <b>backup-policy input</b> into the minimally relaxed half-spaces. Small feasibility tolerances (approximately 2 × 10<super>-5</super>, plus a relaxation tolerance) account for floating-point computations.')
p('Thus “soft safety” has a precise meaning here: use the original inequalities whenever feasible, and relax them only by the amount required when they are inconsistent with actuator limits. The implementation does not trade an arbitrary collision penalty against clarity, and it does not attach an information reward to taking risk. Positive slack can occur without physical contact; zero slack does not certify continuous-time safety.')
h('Candidate acceptance and execution are different checks')
table(['Layer','Behavior in the current mission'],[('Candidate forecast','Reject nonfinite task costs and sampled geometric collisions. Positive QP slack alone does not reject a candidate.'),('Selected-plan check','Evaluate the full 10-second policy without the spatial cache, on the same sampled worlds and all 1,353 cells.'),('Plan failure','If the chosen cost or checked cost is nonfinite, execute the backup through the safety filter for the next period.'),('Actual control','Apply uncached full spline RPCBF at 20 Hz, using only the current real wind belief.'),('Independent audit','Reintegrate saved controls with the physical field at 32 substeps per control interval and inspect geometric clearance.')],[115,CW-115])
p('Report intervention norm ||u<sub>safe</sub> - u<sub>nom</sub>||, common slack, feasibility of the unrelaxed QP, fallback count and physical clearance separately. They describe distinct phenomena. Recorded trajectories are not “certified safe” merely because the solver returned an input.')

# 10
page(10,'Anticipated learning: the objective')
p('At a replanning boundary, the available information is (x<sub>0</sub>, q<sub>0</sub>, <i>B</i><sub>0</sub>). A common candidate vector Y defines the first motion and a causal future policy. Draw S<sub>o</sub> possible wind worlds from <i>B</i><sub>0</sub>. Each is held fixed for its complete candidate rollout.')
eq('saa')
p('In the first period, simulate safety-filtered motion under the current belief and collect synthetic measurements at its executed positions. At the boundary, condition on both observation <b>locations and values</b>. Continue from the actual simulated terminal state and clarity, using that updated belief and the same underlying sampled world. In the current profile, further hypothetical updates occur at 4, 6 and 8 seconds within the 10-second forecast.')
eq('branches')
p('In the original branch formulation, g routes only from earned posterior/state information. Outcomes routed to the same branch share a nominal future sequence. The fast implementation replaces that discrete bank with the shared feedback map described on page 11.', 'small')
h('Where the value of information enters')
p('The first motion determines which observations can be earned. Those observations change the future posterior, nominal wind compensation, posterior-contingent reference and RPCBF filter. These change executed sensing and future clarity cost. MBD can therefore prefer motion that sacrifices some immediate monitoring only when its modeled future consequences repay that loss.')
p('The objective contains <b>only the cumulative monitoring deficit</b> in the current profile. The terminal approximation is zero. There is no entropy bonus, variance-reduction coefficient, positive clipping of an imagined gain, or reset to the original state with information acquired later. A small Monte Carlo objective may still be optimistic because it samples a finite set of outcomes and restricts the future policy class; that optimism is not a formal optimism-in-the-face-of-uncertainty algorithm.')
h('Nonanticipativity')
p('Future controls must be functions of information that would already be available: simulated state, earned measurements and posterior. The same mapping is used in every world. They cannot depend directly on the hidden field or the scenario index. Low uncertainty, by itself, is not evidence of task value; an observation is useful only if a feasible continuation becomes better.')
eq('voi')
p('This same-state diagnostic compares updated and frozen-belief continuations from identical first-stage terminal states and clarity under matched worlds. It should be evaluated with an independent world bank and a declared continuation budget. It is logged separately, may be negative, and is never added to the proposed objective. The current same-state evidence remains weak.','small')

# 11
page(11,'The fast causal policy class')
p('The research formulation uses a shared first-stage sequence and a small bank of future sequences selected by an observation-derived router. That branch formulation is present in the general two-stage implementation. The current demo uses a different, explicitly restricted <b>posterior-feedback continuation</b> to meet the approximately 2-second planning target. It jointly optimizes 18 real coordinates; it does not run an inner MBD optimizer for every hypothetical outcome.')
table(['Coordinates','Physical meaning'],[('1-4','Two 2D acceleration knots for the first period, mapped through tanh to ±u<sub>max</sub> and interpolated.'),('5-12','Four future 2D position knots, mapped through tanh into the workspace.'),('13-18','A shared 2 × 3 gain matrix G that shifts the future reference using posterior summaries.')],[105,CW-105])
eq('features');eq('knots')
p('The posterior summaries are the two mean components and one component standard deviation at the central probe (0, 0). The shift is bounded to 0.6 m per coordinate. In the knot equation, c and h are workspace center and half-width, and the normalized future knot coordinates are η<sub>j</sub>. Knots are additionally clipped 0.08 m inside the workspace boundary.')
p('A five-knot cubic reference starts at the actual first-stage terminal position. Its initial derivative is that terminal velocity and its final derivative is zero. After each hypothetical update, the reference is recomputed with the new posterior but retains the original first-stage terminal anchor and global future-time axis. Only the current and later reference slices are used.')
eq('feedback')
p('The first period uses the acceleration knots in place of the tracking expression. Every nominal input then passes through RPCBF. Both candidate optimization and actual execution use the same declared input limits and drag compensation.')
note('Expressiveness is a research limitation','A single central posterior summary and a bounded reference shift cannot represent every useful observation-dependent route choice. The full belief still informs wind compensation and safety along the path, but it does not enter an unrestricted future optimizer. Policy-class limitations can suppress useful learning even when the inference and MBD code are correct.')

# 12
page(12,'One planning and execution cycle')
p('The following algorithm describes the current 2-second cadence, 10-second horizon profile. Every forecast is causal; only the first 2 seconds are executed before real observations trigger the next plan.')
table(['Step','Operation'],[
 ('1. Real update','At a period boundary, assimilate the last 10 wind observations using full online POAM. Retain the actual state and every cell’s clarity. Freeze the resulting belief for the next execution period.'),
 ('2. Random banks','Draw two outer wind worlds, three independent inner fields and synthetic observation noise. Reuse these banks for all candidates in this planning call.'),
 ('3. Initial candidates','Evaluate seven common geometric initializations: middle and both bypasses toward each region, plus staying. All methods receive the same starts; no alternating schedule is imposed.'),
 ('4. Candidate first stage','Decode Y. For each world, simulate 2 seconds of nominal control, RPCBF filtering, physical propagation, clarity and synthetic sensing under the current belief.'),
 ('5. Candidate continuation','Proposed: condition at 2, 4, 6 and 8 seconds, update the posterior-dependent reference and filter, and continue without resetting state. Current-belief comparator: keep the planning belief frozen throughout the forecast.'),
 ('6. Score and diffuse','Average the complete 10-second task cost over the two worlds. Reject invalid/colliding sampled forecasts. Run four MBD reverse levels with eight proposals each; retain the best evaluated policy.'),
 ('7. Recheck','Re-evaluate the selected policy with uncached fields and all cells. If invalid, set the explicit backup flag.'),
 ('8. Execute','Run the first 2 seconds at 20 Hz with uncached wind-aware RPCBF. The fixed physical field advances the actual state. Store sensing, clarity, controls, slack and clearance.'),
 ('9. Repeat','Update from the real 5 Hz observations at the next boundary. Save numeric checkpoints and timing. Stop at the declared mission duration.')
],[98,CW-98])
h('Boundary conventions matter')
p('The belief used during an actual 2-second period does not assimilate a measurement halfway through that period. Planning synthetic updates use the same boundary convention, at a coarser 0.2-second rollout step. The current-belief comparator still learns from every real period; it omits only imagined future updates within a candidate forecast.')
p('The zero terminal value and 8-second continuation can miss benefits that arrive after the 10-second horizon. Seven common starts cover single transfers and staying, but not complete round trips. These are explicit approximation choices to diagnose, not guarantees of optimal persistent behavior.','small')

# 13
page(13,'Actual model-based diffusion')
p('MBD is a training-free optimizer over the common latent policy vector. It evaluates model rollouts directly and estimates a diffusion score from cost-weighted candidate vectors [R3]. It is neither a pretrained neural diffusion policy nor a single MPPI/CEM update. The simulator and safety filter determine paths; diffusion does not average executed paths into a controller.')
eq('forward')
p('Use four linearly spaced β values from 0.005 to 0.18. Forward-noise the best common initialization, then visit diffusion levels in reverse order. At each level, generate eight Gaussian latent proposals:')
eq('proposal');eq('weights')
p('Here the valid set V contains finite sampled costs. Invalid candidates have weight zero. Subtracting the minimum cost stabilizes exponentiation. The temperature λ = 0.025 is in monitoring-cost units; it controls the cost weighting, not the value assigned to information.')
eq('score')
p('The score is the Gaussian-noise identity evaluated with the weighted candidate mean; the backward update is Monte Carlo score ascent. Proposals remain Gaussian in unconstrained latent space. Tanh mappings inside the simulator enforce physical limits, avoiding a falsely Gaussian clipped proposal distribution.')
h('Selection and failure handling')
p('The best evaluated candidate is retained across all levels, including the initialization. The final denoised latent vector is evaluated separately and replaces that incumbent only if better. If all proposals at a level are invalid, the code uses its explicit neutral-score fallback rather than normalizing NaNs. A fully invalid selected result triggers the mission backup path.')
p('The current budget evaluates seven starts, re-evaluates the selected start inside MBD, evaluates 4 × 8 proposals, and evaluates the final denoised vector: <b>41 candidate objective calls</b>, followed by the uncached selected-plan check. Each wind-aware objective call itself contains multiple worlds and repeated backup filters. CEM and MPPI exist as separately labeled optimizers; the current development mission does not establish that MBD dominates them.')

# 14
page(14,'What denoising shows')
fig('denoising','Actual saved MBD history at t = 20 s of the proposed 120-second mission, seed 8101. Pale paths are the eight candidate rollouts at each reverse level; solid teal is the best evaluated policy and dashed orange is the denoised mean. These paths are forecasts under one sampled world, not observed future flight. The example index was fixed for illustration, not chosen to maximize improvement.',width=CW-50)
table(['Reverse level','Best evaluated cost','Mean candidate cost','Effective sample size'],[
 ('4','0.27339','0.33896','2.88 / 8'),('3','0.25795','0.32267','3.77 / 8'),('2','0.25331','0.26737','6.69 / 8'),('1','0.25331','0.25891','7.96 / 8')],[100,126,138,CW-364])
p('The model score moves the latent search toward low-cost candidates while the noise scale changes. The best sampled cost need not improve at every level, and the final denoised mean need not beat the incumbent. This recorded example improves the incumbent during the middle levels; other cycles retain their initial candidate.')
p('The cost in the table is the sampled 10-second forecast objective. It cannot be compared directly with the 120-second realized mission cost as though they were the same statistic. An attractive denoising animation is evidence that the saved optimization process is visible, not evidence of useful learning or global optimality.','small')

# 15
page(15,'Runtime and computation budget')
p('The expensive operation is repeated belief-aware rollout evaluation: each MBD proposal simulates multiple worlds; each world tick constructs spline RPCBF values and gradients from multiple backup trajectories; future boundaries also condition POAM. A rough workload factor is proposals × worlds × forecast steps × inner fields × backup steps, in addition to kernel and spline computations.')
fig('timing','Measured CPU timing from the retained three 120-second missions. The first two cycles are reported separately because compilation and warm-up dominate. Component medians at right are illustrative and need not sum to the median of the full cycle.')
table(['Method','Median / p95 cycle','Maximum','&gt;2 s'],[('Anticipated learning','1.725 / 1.984 s','2.038 s','2 / 58'),('Current belief','0.771 / 0.826 s','1.080 s','0 / 58'),('Wind-blind nominal + safety','0.067 / 0.076 s','0.091 s','0 / 58')],[173,142,82,CW-397])
p('A measured planning cycle includes generation/input preparation, cache construction, optimization, the full selected-policy check, and the preceding real POAM update and belief snapshot. For proposed, the first two cycles were 25.84 s and 3.11 s; initial POAM setup and snapshot added about 3.19 s separately. The first two comparator cycles were 16.02 s and 0.88 s.')
h('How the budget is achieved')
p('The planner uses JIT-compiled scans, candidate chunk size 2, cached differentiable field evaluation, 18 policy coordinates, two outer worlds, three inner fields and 0.2-second forecast ticks. It omits zero-target cells from candidate scoring with exact original-grid normalization. Actual execution and verification retain all cells and uncached RPCBF.')
p('These are M3 CPU measurements with JAX 0.4.38. Tested Metal configurations failed required operations or host compatibility; no GPU speedup is claimed. The approximately 2-second target is a soft measured planning budget with misses, not an asynchronous real-time flight-controller guarantee. The simulation can pause while planning; deployed latency compensation remains future work.','small')

# 16
page(16,'Baselines and what they isolate')
table(['Method','Nominal planning','Future learning model','Execution safety'],[
 ('Proposed','Wind-aware; RPCBF inside rollout','Hypothetical POAM updates and posterior feedback','Real POAM + RPCBF'),
 ('Current belief','Wind-aware; RPCBF inside rollout','Freeze belief inside forecast; assimilate all real data','Real POAM + RPCBF'),
 ('Wind-blind nominal + safety','Still-air nominal dynamics; no RPCBF in objective','No wind belief in nominal planner','Real POAM + RPCBF'),
 ('Wind omitted from safety','Matched fixed nominal policy in existing safety study','Real POAM continues; nominal wind compensation retained','Zero wind mean and fields inside RPCBF only'),
 ('Oracle diagnostic','Known physical wind explicitly provided','No uncertainty to learn','Privileged reference; not fair baseline')
],[101,148,140,CW-389])
h('The primary anticipation comparison')
p('Proposed versus current belief tests anticipation of future observations. Both arms estimate wind and use it in nominal compensation and actual safety. The comparator is not “unaware of wind.” Differences are caused by optimizing under different future belief evolution, together with the resulting closed-loop histories and stochastic optimization error.')
h('The user-requested nominal comparison')
p('The additional wind-blind planner accepts no wind-belief input when choosing a nominal trajectory. It includes known drag but assumes d = 0 and does not model the execution safety filter in its objective. The actual wind-aware RPCBF still protects its execution. This contrast answers whether wind-aware, filter-aware planning helps relative to planning first and handling wind only at execution. It does not isolate anticipated learning.')
h('Fairness and additional diagnostics')
p('The three missions share task geometry, initial state and clarity, the physical field, historical survey, observation-noise sequence, horizon, actuator limits and nominal MBD proposal count. Their paths differ, so earned observation positions differ. Equal proposal counts are not equal runtime: a wind-aware candidate is much more expensive.')
p('A nominal-planner sensitivity increased proposals from 8 to 256 per reverse level. Its task cost improved only about 0.97%, but its runtime remained below the proposed planner. An unintended chunk-size change (2 to 1) was retained and separately checked. This is useful optimizer evidence, not a strict one-parameter or equal-wall-time comparison. Weighted information gain, heuristic, CEM/MPPI and older branch variants exist elsewhere in the repository; their historical results must not be presented as this current three-arm experiment.')

# 17
page(17,'Reproducible development configuration')
p('The following values describe the retained 120-second comparison, not generic defaults from the configuration dataclass. The driver overrides the base YAML. In particular, the task-specific obstacle radius is 0.50 m even though a generic configuration field contains a different value.')
table(['Component','Value in this recording'],[
 ('Mission','120 s; 60 replans; seed 8101; start at A with zero velocity.'),
 ('Geometry','Bounds [-2.7, 2.7] × [-2.1, 2.1] m; A = (1.5, 0.9), B = (-1.5, -0.9). Each region is 1.4 × 1.2 m.'),
 ('Obstacles','Centers approximately (-0.37044, 0.61739) and (0.37044, -0.61739) m; radius 0.50 m; filter margin 0.06 m.'),
 ('QoI grid','41 × 33 = 1,353 cells; 90 cells per monitored region; target 0.70 in A/B and zero outside; initial q = 0.74 everywhere.'),
 ('QoI dynamics','Uniform Q = 0.16; R = 0.10; Gaussian sensor width σ<sub>s</sub> = 0.50 m. General code supports spatial Q; this profile is uniform.'),
 ('Dynamics','γ = 1; component control bounds ±1.6 m/s²; planar RK4.'),
 ('Forecast and execution','10 s forecast = 2 s first stage + 8 s continuation; forecast dt = 0.20 s; execution dt = 0.05 s.'),
 ('Wind observations','σ<sub>d</sub> = 0.10 m/s²; 5 Hz observations; 2 s batch updates; identical 42-point historical survey.'),
 ('POAM','24 inducing inputs; 4 attentive length scales; 10 real kernel-training steps/update; hypothetical relocation enabled; kernel frozen inside planning.'),
 ('Posterior sampling','32 Sobol spectral nodes; 667 dense prior support points; 2 outer worlds; 3 independent safety fields.'),
 ('RPCBF','Full not-a-knot spline; 8 backup steps × 0.15 s; backup gain 2; α = 1; actuator box hard, barrier slack on infeasibility.'),
 ('MBD','18 latents; 7 shared starts; 4 reverse levels; 8 proposals/level; β = 0.005 to 0.18; λ = 0.025; chunk size 2.'),
 ('Task objective','Mean positive-part clarity deficit normalized by 1,353 cells; zero terminal term; no information reward.'),
 ('Checks','Uncached selected-policy check; physical dense collision audit at 32 substeps per 0.05-second executed input.')
],[122,CW-122])
p('Several legacy dataclass fields are not consumed by this compact planner. In particular, the presence of an ig_weight value in a saved config does not mean the proposed objective uses it. Treat the invoked driver and evaluator, not the list of available fields, as the definition of the method.','small')

# 18
page(18,'Recorded persistent behavior')
fig('missions','Full retained 120-second executions, seed 8101. Left: actual planar trajectories. Right: mean clarity in each large region; the dashed horizontal target is 0.70. Region A/B line styles alternate as the robot revisits. All plotted clarity values come from executed sensing, including periods when the other region loses information.',width=CW)
table(['Method','Mission cost','Transfers','Fallbacks','Dense contacts'],[('Proposed','2.83933','16','0','0'),('Current belief','2.95491','14','1','0'),('Wind-blind nominal + safety','3.49865','8','5','0')],[174,88,70,79,CW-411])
p('The current mission demonstrates repeated monitoring, rapid clarity loss and recovery, and visible differences in route selection. It does not show a clean, consistently learned “middle outward / bypass return” policy: proposed uses the middle for one of its 16 recorded transfers, and the comparator also discovers middle transfers. The intended mechanism should be judged by costs and causal diagnostics, not by assigning meaning to a single attractive path.','small')

# 19
page(19,'Whole-field clarity and task cost')
fig('clarity_maps','Proposed execution at predetermined times. All 1,353 cells evolve with Q; white segments show the preceding 8 seconds of motion and red dots the current vehicle. A/B rectangles mark positive targets. Background clarity decays too, although its target is zero. The common color scale makes decay and local recovery comparable.')
fig('cost','Recorded integrated cost and instantaneous deficit. The normalization is the whole 1,353-cell grid. Since only 180 cells have positive targets, the integrated ROI-normalized costs are larger by 1,353/180; this scaling leaves paired percentage differences unchanged.',width=CW)
p('<b>Development contrasts:</b> proposed is 3.91% lower cost than current belief, and 18.84% lower than wind-blind nominal planning with execution safety. The first is the relevant isolated anticipation comparison; the second combines awareness of wind, awareness of filter interventions and anticipation. One selected development seed supports neither a confidence interval nor a general performance claim.','small')

# 20
page(20,'Wind estimation and calibration')
fig('wind','Full POAM snapshots from the proposed mission, shown against hidden truth only for retrospective diagnosis. Top: disturbance-vector means and physical field. Bottom: component standard deviations and final component RMSE at each cell. Small posterior variance does not by itself imply accurate or calibrated predictions.')
p('The independent mission audit reports passage RMSE falling from 0.9166 to 0.1528 m/s² for proposed, with mean component standard deviation falling from 0.4894 to 0.0433 m/s². Current belief also learns: its final passage RMSE is 0.1588 and standard deviation 0.0471. These numbers show learning in <b>both</b> arms; the audit passage mask differs from the separate calibration study mask.')
h('A remaining scientific problem: undercoverage')
p('A separate passage-grid calibration diagnostic found only about 31.9% / 43.3% of component values inside nominal ±1.96σ intervals at 120 s (proposed / current belief). A dense GP using the same learned kernel also undercovered, about 32.4% / 23.8%. Sparse capacity or statistic transport alone therefore does not explain the problem. Spatial grid points are correlated; these are diagnostic coverage fractions, not independent confidence trials.')
p('A declared coarse forecast mean and a half-length-scale prior sensitivity were retained. The latter improved diagnostic coverage to about 69% in both histories but did not establish calibrated uncertainty. Neither diagnostic was promoted into the production planner. Posterior-driven safety and anticipated learning remain sensitive to model misspecification; a better-looking mean map is insufficient evidence of reliability.','small')

# 21
page(21,'Safety evidence and research status')
p('The three current 120-second missions have no contacts in independent dense reintegration. Minimum geometric clearances were 0.08991 m (proposed), 0.12724 m (current belief) and 0.11087 m (wind-blind nominal + safety). This establishes those recorded outcomes, not a general collision-rate guarantee.')
h('What the separate wind-safety study shows')
table(['Declared study','Outcome / interpretation'],[('24 fresh fixed-policy seed pairs','1 collision with wind-aware filtering, 2 with wind omitted from filtering. Only one discordant pair; exact two-sided paired p = 1.0.'),('Mixed environment, fresh seed 634','Aware clearance +0.12176 m; wind-omitted clearance -0.01355 m. A physically checked ablation-only contact example.'),('Shear environment, fresh seed 623','Aware -0.09368 m; omitted -0.12796 m. Both filters collide; actuator saturation and positive slack were observed.'),('Three positive controls','Original development seeds, retained separately and excluded from fresh-seed inference.')],[182,CW-182])
p('This study replays the same nominal waypoint policy. It keeps nominal POAM wind compensation in both arms and removes wind only from RPCBF. Each arm updates POAM using its own real observations. It is a fixed-policy safety ablation, not a comparison of two fully replanned policies. After-contact motion is a diagnostic continuation without collision response. The collision audit uses 32 physical RK4 substeps per saved input.')
h('What has and has not been demonstrated')
table(['Question','Current status'],[('Can the implementation run a persistent mission?','Yes: full 120-second records, repeated transfers, full-grid clarity, POAM and saved MBD histories.'),('Can it generate a 10-second forecast near 2 seconds?','Yes for the measured warmed CPU profile, with 2/58 misses; startup is separate.'),('Is anticipation substantially better?','Not yet: 3.9% on one development seed and weak same-state evidence.'),('Is regret limited when learning has little value?','Not established in the final matched full-POAM protocol.'),('Does ignoring wind in safety cause collisions?','A matched physical example exists; a reliable collision-rate advantage or replanned contrast is not established.')],[212,CW-212])
p('The original three scientific acceptance cases therefore remain open. Software correctness, faster computation and attractive visualizations are necessary supporting evidence; they are not substitutes for the missing learning-benefit and negative-control studies.','small')

# 22
page(22,'Verification and bounded next steps')
p('Existing component tests target mathematical identities and information flow. This report uses the retained evidence and data; its generation does not rerun the heavy mission studies. Earlier numerical failures and unfavorable results remain part of the provenance.')
table(['Area','Relevant checks / test modules'],[('Clarity and dynamics','Riccati/clarity behavior, decay and sensing, integration convergence, drag consistency: test_clarity, test_field_monitoring, test_drag_dynamics.'),('POAM and sampling','Full residual covariance, real/hypothetical conditioning, independent matrix algebra, sample moments and derivative checks: test_poam, test_hypothetical_poam, test_wind_posterior.'),('RPCBF and QP','Spline extrema, finite-difference value gradients, independent 2D constrained solves, truth separation: test_rpcbf, test_safety_behavior.'),('MBD','Score identity, noise schedule, nonconvex/analytic objectives and optimizer separation: test_mbd.'),('Causality and policy','Terminal-state continuation, value-dependent updates and common policy structure: test_two_stage, test_continuation, test_budgeted.'),('Runtime / ablation','Cache comparison, zero-wind nominal API, matched safety inputs: test_runtime_cache, test_wind_blind_planner, test_wind_safety_ablation.'),('Recorded missions','All-cell cost reconstruction, dense physical contacts, startup and steady timing, failed selected checks retained in audit JSON.')],[126,CW-126])
h('Research sequence before claiming useful VoI')
p('<b>1. Diagnose calibration.</b> Use prequential next-batch predictive scores and independently declared wind families, alongside spatial coverage. Separate kernel misspecification, observation geometry and sparse approximation. Do not tune a prior solely to obtain favorable hidden-truth comparisons.')
p('<b>2. Test the causal mechanism at fixed states.</b> Compare updated and frozen-belief continuations from the same state and clarity with independent random banks. Vary the continuation horizon and policy expressiveness. Add shared round-trip initializations as an optimizer diagnostic, without imposing alternation.')
p('<b>3. Separate approximation and optimizer error.</b> Increase world/inner counts in offline audits, compare cached and uncached rankings, and compare MBD with CEM/MPPI under declared rollout and wall-time budgets. Preserve all regressions.')
p('<b>4. Freeze a design, then evaluate fresh seeds.</b> The declared positive target is at least 10% mean paired cost reduction with a 95% interval excluding zero. The low-value target is a one-sided 5% noninferiority margin. Use at least eight fresh paired seeds per regime; run matched replanned safety ablations and report every contact and failure. These are planned gates, not completed results.','small')

# 23
page(23,'Reproduction and source provenance')
p('The repository is self-contained: implementation, environments and results live in voi-diffusion-ipp. Sibling reference repositories were inspected for methods and architecture but are not runtime imports. Source HEAD for this report is <b>3a31937d4924b92af9bb6cec03a2c761fe4d86e8</b>; local visualization edits were present. Each mission retains its own executed source/driver hashes, which take precedence over later HEAD for replay.')
h('Reproduce the current development pair')
p('Run from the repository root in its UV-managed environment. Choose a new output directory: the driver intentionally rejects an existing directory. These commands are supplied for reproduction and were not executed to generate this report. Run only one heavy worker on the 16 GB M3.','small')
code(['JAX_PLATFORMS=cpu uv run --no-sync python scripts/run_budgeted_mission.py \\', '  --periods 60 --seed 8101 --first-stage 2 --future-update-steps 10 \\', '  --relocate --rapid --chunk 2 --prior-regime outer-survey \\', '  --methods proposed current_belief \\', '  --output results/budgeted_missions/reproduction_new'])
p('For the additional nominal comparison, use the same flags and a separate new directory with <b>--methods wind_blind_planner</b>. Auditing/export scripts are scripts/audit_budgeted_mission.py and scripts/export_budgeted_demo.py; inspect their --help for required input paths. Use uv sync first only when reconstructing the pinned environment. The validated execution backend is native ARM64 CPU; Linux/CUDA portability is intended but not measured on this host.','small')
table(['Implementation','Location relative to repository root'],[('QoI / physics','src/voi_diffusion_ipp/clarity.py; dynamics.py; experiments/two_region.py'),('POAM / sampler / cache','src/voi_diffusion_ipp/wind/poam.py; poam_belief.py; cache.py'),('Safety','src/voi_diffusion_ipp/safety/rpcbf.py; spline.py; qp.py'),('Current planning','src/voi_diffusion_ipp/planning/budgeted.py; mbd.py; wind_blind.py'),('General two-stage variant','src/voi_diffusion_ipp/planning/two_stage.py; rollout.py'),('Current mission pair','results/budgeted_missions/persistent_two_region_2s_01/'),('Nominal comparator','results/budgeted_missions/persistent_two_region_safety_only_01/'),('Protocols / findings','docs/two_region_design/; docs/wind_safety_validation/; docs/persistent_research_worklog.md')],[135,CW-135])
p('The two interactive views replay the same archived numerical results. Open <link href="http://127.0.0.1:8765/?run=budgeted_persistent_two_region_2s_01_proposed_8101&amp;baseline=wind_blind_planner" color="#087e8b">the 2D scientific demo</link> or <link href="http://127.0.0.1:8765/3d/?run=budgeted_persistent_two_region_2s_01_proposed_8101&amp;baseline=wind_blind_planner" color="#087e8b">the 3D scene</link> while the local demo server is running. The report build scripts and SHA-256 manifest are retained in docs/method_report/.','small')

# 24
page(24,'References and attribution')
refs=[
 ('R1','Supplied research discussion, <i>Disturbance Aware IPP Purpose</i>, 22 pages, especially pp. 7-13. Internal design discussion; not a peer-reviewed publication. Workspace source: codex_instructions/Disturbance Aware IPP Purpose.pdf.'),
 ('R2','K. Ben Naveed, <i>Informative Planning with Task-Aware Spatial Disturbance Estimation</i>, supplied 9-page research draft. Workspace source: papers/Disturbance_aware_ipp.pdf. Read with CODEX_IMPLEMENTATION_INSTRUCTIONS_MAC.md and subsequent declared model revisions.'),
 ('R3','C. Pan, Z. Yi, G. Shi and G. Qu. <i>Model-Based Diffusion for Trajectory Optimization.</i> NeurIPS 2024. Core score/proposal/reverse update: supplied paper Eqs. 3, 6, 8-9, Algorithms 1-2. <link href="https://lecar-lab.github.io/mbd/" color="#087e8b">Author project and paper</link>.'),
 ('R4','L. Knoedler, O. So, J. Yin, M. Black, Z. Serlin, P. Tsiotras, J. Alonso-Mora and C. Fan. <i>Safety on the Fly: Constructing Robust Safety Filters via Policy Control Barrier Functions at Runtime.</i> IEEE Robotics and Automation Letters 10(10), 2025. Sec. III C-D, Eqs. 14-21. <link href="https://doi.org/10.1109/LRA.2025.3597847" color="#087e8b">doi:10.1109/LRA.2025.3597847</link>.'),
 ('R5','K. Ben Naveed, U. Sahai, A. Girard and D. Panagou. <i>Provably Safe Stein Variational Clarity-Aware Informative Planning.</i> Supplied L4DC 2026 manuscript / arXiv:2511.09836. Clarity/Riccati dynamics are used; the Stein optimizer and gatekeeper safety theorem are not transplanted. <link href="https://arxiv.org/abs/2511.09836" color="#087e8b">Manuscript</link>.'),
 ('R6','W. Chen, L. Liu and R. Khardon. <i>POAM: Probabilistic Online Attentive Mapping for Efficient Robotic Information Gathering.</i> Robotics: Science and Systems, 2024. Attentive kernel, inducing selection, statistic transport and variational EM. <link href="https://www.roboticsproceedings.org/rss20/p019.html" color="#087e8b">doi:10.15607/RSS.2024.XX.019</link>.'),
 ('R7','M. Faessler, A. Franchi and D. Scaramuzza. <i>Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories.</i> IEEE Robotics and Automation Letters, 2018. Motivation for drag; this report uses a simpler planar model. <link href="https://arxiv.org/abs/1712.02402" color="#087e8b">arXiv:1712.02402</link>.'),
 ('R8','J. Wilson, V. Borovitskiy, A. Terenin, P. Mostowsky and M. Deisenroth. <i>Efficiently sampling functions from Gaussian process posteriors.</i> ICML, PMLR 119:10292-10302, 2020. Decoupled posterior sampling. <link href="https://proceedings.mlr.press/v119/wilson20a.html" color="#087e8b">Paper and proceedings</link>.'),
 ('R9','E. Klenske and P. Hennig. <i>Dual Control for Approximate Bayesian Reinforcement Learning.</i> JMLR 17, 2016. Background for control-dependent learning and finite-horizon value. <link href="https://jmlr.org/papers/v17/15-162.html" color="#087e8b">JMLR article</link>. No dual-control theorem is claimed for this implementation.'),
]
for tag,txt in refs:p(f'<b>[{tag}]</b> {txt}','small')
h('Implementation references')
p('Read-only reference repositories: model-based-diffusion (MBD); rpcbf-safety-on-the-fly (RPCBF); safe-mpd (safety-aware candidate evaluation); vg-dip (clarity and POAM reference); mbd-playground (denoising/execution UI organization). Mathematical reimplementations and licensing decisions are recorded in docs/reference_mapping.md and docs/licenses/. This report separates reference-method attribution from the new integration and its empirical claims.','small')

class ReportCanvas(canvas.Canvas):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.states=[]
 def showPage(self):self.states.append(dict(self.__dict__));self._startPage()
 def save(self):
  total=len(self.states)
  for state in self.states:
   self.__dict__.update(state);self.setStrokeColor(colors.HexColor('#d5e0e4'));self.setLineWidth(.5);self.line(M,43,W-M,43)
   self.setFillColor(GRAY);self.setFont('Arial',8);self.drawString(M,29,'VoI Diffusion IPP | Implementation and evidence | 17 Oct 2026');self.drawRightString(W-M,29,f'{self._pageNumber} / {total}')
   canvas.Canvas.showPage(self)
  canvas.Canvas.save(self)

def header(c,doc):
 if doc.page>1:
  c.setFont('Arial',8);c.setFillColor(GRAY);c.drawString(M,H-31,'PERSISTENT MONITORING UNDER LEARNED SPATIAL WIND');c.setStrokeColor(colors.HexColor('#d5e0e4'));c.line(M,H-39,W-M,H-39)

doc=SimpleDocTemplate(str(OUT),pagesize=(W,H),rightMargin=M,leftMargin=M,topMargin=57,bottomMargin=58,title='Value-of-Information Diffusion Planning: Method, Implementation and Evidence',author='voi-diffusion-ipp research workspace',subject='POAM, MBD, RPCBF and persistent clarity monitoring',pageCompression=1)
doc.build(story,onFirstPage=header,onLaterPages=header,canvasmaker=ReportCanvas)
# Preserve link annotations and add a clickable section outline after pagination.
from pypdf import PdfReader, PdfWriter
reader=PdfReader(str(OUT))
assert len(reader.pages)==24, f'Unexpected pagination: {len(reader.pages)}'
writer=PdfWriter();writer.clone_document_from_reader(reader)
for num,title in sections:
 writer.add_outline_item(title or 'Value-of-Information Diffusion Planning',num-1)
with OUT.open('wb') as handle:writer.write(handle)
manifest={'title':'Value-of-Information Diffusion Planning','date':'2026-10-17','source_head':'3a31937d4924b92af9bb6cec03a2c761fe4d86e8','scope':'Documentation only; no research jobs or automations started','sections':sections,'equation_count':eqno,'pdf_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'figures':json.loads((ASSET/'figure_manifest.json').read_text())}
(ROOT/'docs/method_report/provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT)
