## Reviewer dMEa

### do not have enough experiments and unclear contribution

```
Choose The Contribution Type You Are Using For Your Review: Algorithms/General
Justify The Above Choice:
I agree

Paper Summary:
This paper introduces the framework in which the robot can handle various complex interactive scenarios. There are two LLMs presented – one LLM serves as a main controller for the current task, and the other serves as a standby model that listens to the user command. The paper also trains another model called action expert for motor control. The proposed method is verified in Libero benchmark. One demo is shown in the real robot.

Paper Strengths:
The paper in general is easy to understand and follow.

Major Weaknesses:
There are a couple of shortcomings:

There is no real world benchmark presented in the paper, which makes it hard to judge the correctness of the approach, and show real world impact.
Using Libero benchmark to evaluate the model is insufficient as this model is already saturated. The authors are encouraged to use more challenging benchmarks.
The contribution of the paper is unclear. It seems that this is just a combination of couple of existing systems and show a demo.
Minor Weaknesses:
See above

Preliminary Recommendation: 2: Weak Reject
Justification Of Preliminary Recommendation:
No significant contribution to the community, and see the weakness section.

Suggestions For Rebuttal:
Try to address the weakness.

Ethics Review Flag: No
Confidence Level: 4: High Confidence - The reviewer has strong expertise in the area. They are highly familiar with the relevant literature and can critically evaluate the paper.
```

## Reviewer PgnX

### review

```
Choose The Contribution Type You Are Using For Your Review: Applied/Systems
Justify The Above Choice:
I don't think it's a work about algorithms though the authors claimed it, in my opinion it's more like an engineer system design for audio-related robot.

Paper Summary:
This paper introduces VITA-E, a hierarchical system with VLM and VLA for natural, concurrent, and interruptible human-robot interaction. The core innovation is the model-as-controller paradigm, which fine-tunes a vision-language model (VLM) to generate special control tokens such as [RES], [ACT], [HALT], and [END] to directly control action expert states. VITA-E also adopts a dual-model architecture consisting of two identical full VLA instances: an Active model for ongoing task execution and a Standby model for continuous listening and preemption control. Evaluations on a humanoid robot show that VITA-E achieves 100% success rates in voice interruption and emergency stop, and 93.3% in task switching, while maintaining competitive manipulation performance.

Paper Strengths:
The work present several interesting engineering designs that enable interaction capabilities with only simple modifications to the current hierarchical architecture, including the use of special tokens to mark operation states and the deployment of two identical systems to achieve preemption and concurrent processing.

Major Weaknesses:
This paper is closer to an engineering effort, focusing on engineering design optimizations for interaction, with limited core algorithmic contributions. The dual-model architecture, where a VLM handles planning and a VLA handles execution, is very common. The introduction of special tokens (e.g., [ACT], [HALT], [RES]) is quite intuitive and hardly qualifies as a core innovation. The use of active and standby models to achieve concurrency, switching, and preemption is also highly engineering-driven, lacking substantial algorithmic novelty.

The experimental design for manipulation is questionable. Given the hierarchical architecture, it is unclear why the action expert did not adopt the best-performing model (e.g., π-0.5). Replacing GroNNT’s backbone with VITA-1.5 leads to performance degradation and results below π-0.5. What is the rationale behind this design choice?

The core focus of this paper, interaction experiments, is very unclear and limited. Experiments are conducted only on a small set of tasks, and the evaluation methodology is ambiguous (e.g., does the success rate in line 338 include successful execution of the manipulation?).

There are already several end-to-end works on embodied interaction (e.g., ELLSA [1], RoboOmni [2]), yet the authors do not compare their approach with these works or discuss the differences and advantages.

[1] End-to-end Listen, Look, Speak and Act, ICLR 2026.
[2] RoboOmni: Proactive Robot Manipulation in Omni-modal Context, ICLR 2026.

Minor Weaknesses:
The latency of interaction is discussed only qualitatively, lacking quantitative data analysis. Additionally, there is no quantitative data on GPU usage.

All experiments are conducted on a single robotic platform and within a single scenario (pnp table), lacking validation of generalization across different platforms and tasks.

Preliminary Recommendation: 2: Weak Reject
Justification Of Preliminary Recommendation:
The paper presents itself as an algorithmic contribution, but is more accurately an engineering system integration: the hierarchical architecture is already issued in this filed and the special token mechanism is straightforward, leaving limited algorithmic novelty. Also, the empirical support is undermined by limited experiments and questionable setting.

Suggestions For Rebuttal:
see weakness above

Ethics Review Flag: No
Confidence Level: 4: High Confidence - The reviewer has strong expertise in the area. They are highly familiar with the relevant literature and can critically evaluate the paper.
```

## Reviewer XANf

### Promising system design but insufficient empirical validation

```
Choose The Contribution Type You Are Using For Your Review: Applied/Systems
Justify The Above Choice:
The contribution in this paper is mainly a dual-model framework. It is better captured as system engineering rather than an algorithms/methods paper.

Paper Summary:
The paper presents VITA-E, a framework for human-robot interaction that enables concurrent perception, speech, and action, along with real-time interruption handling. The main idea is a dual-model architecture consisting of an Active model for execution and a Standby model for continuous listening and intervention. The system also introduces a “model-as-controller” paradigm, where a vision-language model generates special tokens (e.g., [ACT], [HALT]) that directly control system behavior. Experiments on simulation benchmarks and a real humanoid robot demonstrate strong performance on interaction tasks such as interruption, task switching, and emergency stopping, while maintaining acceptable manipulation ability.

Paper Strengths:
Addresses an important and under-explored problem: real-time, concurrent, interruptible interaction is a key missing capability in current mainstream VLA systems.
Novel and intuitive system design: The dual-model architecture is a clean and practical solution to concurrency and interruption.
Broad compatibility: The framework is designed to be compatible with various VLA models, enhancing its potential impact on the field.
Major Weaknesses:
Computational overhead of running two parallel VLA model instances as "hemispheres", which may limit deployment on edge robotic hardware.
Lack of strong interactive baselines: Interactive capabilities are evaluated almost exclusively on VITA-E, making it difficult to quantify gains over prior or simpler systems.
Manipulation performance gap: The method underperforms prior work (e.g., GR00T on Libero), raising concerns about trade-offs between interactivity and core task capability.
Limited task diversity: Real-robot experiments cover only a small set of simple tasks, insufficient to support claims about general, open-ended interaction.
Minor Weaknesses:
The concurrency task is evaluated only qualitatively; a quantitative metric (e.g., degradation in action success rate while speaking vs. without speaking) would strengthen the claim.
Lacks discussion of robustness to noisy or ambiguous speech inputs. (e.g., the difference between "No stop." and "No. Stop!" shown in the demo video)
Preliminary Recommendation: 3: Borderline Reject
Justification Of Preliminary Recommendation:
This paper tackles a highly relevant problem in embodied AI systems: enabling fluid, real-time, interruptible human-robot interaction, which is essential for practical deployment. The proposed dual-model architecture and model-as-controller paradigm are intuitive, well-motivated, and practically useful.

However, the work is best characterized as a system and integration contribution rather than a fundamental advance in modeling. The empirical evaluation is the main limitation: the lack of strong baselines for interaction, limited task diversity, small-scale experiments, and missing statistical rigor weaken the strength of the claims. Additionally, the performance gap in manipulation and unquantified computational cost raise concerns about trade-offs and scalability.

Suggestions For Rebuttal:
Quantify the dual-model computational overhead, e.g., GPU memory usage and inference latency figures comparing VITA-E to the single-model baselines. Elaborate on how the system handles "false positive" interruptions, such as when a user is talking to someone else in the room rather than the robot. More baselines for comparison: Compare the dual-model approach against a single-model baseline that uses a more frequent inference clock or "look-ahead" tokens to see if the second model instance is strictly necessary for all scenarios.

Ethics Review Flag: No
Confidence Level: 4: High Confidence - The reviewer has strong expertise in the area. They are highly familiar with the relevant literature and can critically evaluate the paper.
```
