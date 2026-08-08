# C.3 Computational Mechanics

[**Contributors	1**](#contributors)

[**Module intent	2**](#module-intent)

[**Prerequisites	3**](https://docs.google.com/document/d/1rHhFUrGKpFHXZTU9AHeWd-fvmHpjU2j1EpFPEbATgQ8/edit?tab=t.8kuemdcvmj3e#heading=h.lmuattl3ymdf)

[**Content	3**](#content)

[Fast track	3](#fast-track)

[Main content	3](#main-content)

[Learn more	5](#learn-more)

[**Teaching guide	6**](#teaching-guide)

[Time Schedule	6](#time-schedule)

[Required materials	6](#required-materials)

[\[Sub-module 1..n\]	7](https://docs.google.com/document/d/1rHhFUrGKpFHXZTU9AHeWd-fvmHpjU2j1EpFPEbATgQ8/edit?tab=t.8kuemdcvmj3e#heading=h.hvuzaaydm9rk)

[**Postprocessing	7**](#postprocessing)

# Contributors {#contributors}

Xavier Poncini (Simplex) wrote and taught all sections.

# Module intent {#module-intent}

**What.** At a high-level, the goal of this module is to present computational mechanics as a research program that seeks to anticipate structure in the internal representations of a near-optimal predictor (e.g., a neural network) by considering (a) the stochastic process it is predicting, and (b) the architecture of the predictor. The students should be able to:

* Explain the role of the near-optimal assumption in constraining the representations relevant to the predictor.   
* Outline the limits of the research program i.e., for what classes of stochastic processes & predictor architectures have we anticipated representations for. 

The module then gives students a sense of the low-level details by fixing the predictor to be a transformer and the stochastic process to be such that it admits a hidden Markov Model (HMM) realisation. The students should be able to:

* Recall the definition of HMMs and generalised HMMs (GHMMs)  
* Compute the belief state of a GHMM and explain why it is a useful object for making predictions about GHMM data.  
* Explain the evidence for the claim: transformers represent belief state geometry in their residual stream. 

**Why.** Optimal prediction is a natural instrumental subgoal of goal-directed systems. As such, we expect AI systems to become increasingly well-approximated by optimal predictors, and thus develop representations that facilitate these predictions. Understanding and anticipating representations relevant to such systems will provide affordances beyond those that are currently surfaced by behavioural analysis. 

**How.** The high-level overview is introduced explicitly in the first lecture, with similar themes recurring throughout the content. The following lectures and exercise sessions introduce HMMs & GHMMs, and the minimal sufficient statistics for prediction – the belief state. The reading session explores an example of identifying the belief states in a transformer trained on HMM data. In the workshop, the students design their own HMMs to satisfy various desirable properties. 

# Prerequisites

**Mechanistic Interpretability.** Familiarity with the goals and basic methodology of mechanistic interpretability, including the notions of features and circuits; the linear representation hypothesis; linear probes as a method for reading off internal representations; and a basic understanding of the transformer architecture. Students should be comfortable with the idea that one can train a linear map from model activations to some target structure and evaluate its quality (e.g. via MSE or R²).

**Mathematical background.** The module is mathematically self-contained – all formal definitions (HMMs, GHMMs, belief states, MSPs) are introduced from scratch. However, students will engage with the material more fluently if they are comfortable with the following:

* **Linear algebra:** row-stochastic matrices, rank and invertibility, null spaces (kernels).   
* **Probability:** conditional probability, Bayes' rule, probability distributions over finite sets, the probability simplex.   
* **Basic machine learning:** next-token prediction, loss functions (cross-entropy), the concept of model activations.

# Content {#content}

## Fast track {#fast-track}

Complete the main reading from the reading session. Focus on being able to answer the following questions:

* What is a hidden Markov model (HMM)?  
* What is a belief state?  
* What is the mixed state presentation (MSP)?  
* About the map learnt from activations to belief states:   
  * What is the source?   
  * What is the target?   
  * How is the quality of the map evaluated?  
* Explain each element of the following diagram:

![][image1]

## Main content {#main-content}

### Overview & Scope 

*Intent:* To give a high-level overview of the research program of applying computational mechanics to understand the internal structure of near-optimal predictors (mirroring the module intent above). Give the students an overview of the day to come. 

*Teaching notes ([slides](https://14xp.github.io/assets/slides/CompMechSlidesSummer26_Pt1.pdf)):*

* Slide 13 contains a diagram that relates the theoretical framework (belief state) to identifying structure in structure in transformers, highlighting that this is the key diagram to understand and we will return to it in the reading session. 

### Computational Mechanics Foundations 

#### Lecture: Hidden Markov Models 

*Intent:* To introduce *hidden Markov models* (HMMs), and consider stochastic processes that admit realisations in terms of HMMs. By considering optimal predictors with a *simplicity-bias* (e.g., favouring low-dimensional representations), we motivate the notion of *minimal* HMM realisations i.e., those that have the fewest number of hidden states. We then highlight that a stochastic process (admitting an HMM realisation) need not have a unique minimal HMM realisation. This observation motivates the definition of *generalised hidden Markov models* (GHMMs), which relax some conditions of HMMs, in service of always admitting a unique GHMM realisation.

*Teaching notes ([slides](https://14xp.github.io/assets/slides/CompMechSlidesSummer26_Pt2.pdf)):*

* Present the intuitive introduction to HMMs, followed by the formal one.  
* Take time to go through the examples making sure the students follow how the diagrams correspond to matrix elements.

#### Exercises: Hidden Markov Models 

*Intent:* To give the students hands-on experience implementing HMMs with Python. Following ARENA-style exercise sessions, students will implement class methods to:

* Validate whether input transition matrices and initial states correspond to an HMM.  
* Compute the probability of an input sequence.  
* Compute the next-token probability (NTP) conditional on an input sequence.   
* Visualise the NTP geometry. 

*Teaching notes ([exercises](https://colab.research.google.com/github/14xp/iliad-comp-mech-materials/blob/main/exercises/colab/part1_sequence_probabilities_exercises_colab.ipynb) and [solutions](https://colab.research.google.com/github/14xp/iliad-comp-mech-materials/blob/main/exercises/colab/part1_sequence_probabilities_solutions_colab.ipynb)):* 

* Hover around the class, and be available to give hints when students are stuck.    
* For students that have been stuck for longer than the expected exercise time, encourage them to look at the hint / solution.

#### Lecture: Prediction

*Intent:* To demonstrate that the *belief state* is the minimal sufficient statistic for predicting emissions from HMMs. We demonstrate that this object is instrumentally useful for making future predictions by showing that there exists a multilinear map from the belief state to the distribution over future token emissions. Then, we introduce the *mixed state presentation* (MSP) as the natural inference process over belief states, and explore the MSP of the *zero-one-random* (z1r) process in detail. We conclude by generalising this machinery to GHMMs. 

*Teaching notes ([slides](https://14xp.github.io/assets/slides/CompMechSlidesSummer26_Pt3.pdf)):*

* The core intuition is developed in the z1r example, take the time to explain this in detail.  
* Slide 21 highlights how each of the pieces: MSP, belief state and next-token prediction; fit together. Linger on this slide to ensure the students appreciate the connections.   
* If short on time, skip the GHMM generalisation. 

#### Exercises: Prediction

*Intent:* To give the students hands-on experience computing and visualising belief states with Python. Following ARENA-style exercise sessions, students will implement class methods to:

* Compute the belief state corresponding to an input sequence.  
* Perform a belief update given a past belief and a new observation.  
* Compute the NTP associated with a belief state.  
* Compute all belief states for sequences up to a fixed depth.   
* Visualise the belief geometry

*Teaching notes ([exercises](https://colab.research.google.com/github/14xp/iliad-comp-mech-materials/blob/main/exercises/colab/part2_belief_states_exercises_colab.ipynb) and [solutions](https://colab.research.google.com/github/14xp/iliad-comp-mech-materials/blob/main/exercises/colab/part2_belief_states_solutions_colab.ipynb)):* As in the previous exercise session.

### Applications

#### Reading Session: Belief Geometry In Transformers

*Intent:* To engage with literature providing empirical evidence that neural networks (e.g., transformers & RNNs) trained on data emitted from (G)HMMs represent the belief-state geometry in their activations. After the reading, students will participate in a small-group discussion about the strengths and weaknesses of the evidence provided in the papers.

*Teaching notes ([reading guide](https://docs.google.com/document/d/1QbtXABwcpbmz_Wts4pdbdtQUTOCf_rWj-_2-1melRzc/edit?usp=sharing)):*

* Make the students aware of the reading guide.  
* Hover around the class, and be available to answer questions.  
* Give time warnings about when the discussion period will begin.

#### Workshop: Designing Processes Worth Studying

*Intent:* To give students the opportunity to design their own HMM processes, possibly endowed with the following properties:

* Belief structure is richer than NTPs.  
* Easy to visualise belief and NTP geometry.  
* Striking belief geometry.   
* Related to properties of natural data. 

Have the students present their processes to the class. 

*Teaching notes ([workshop sheet](https://docs.google.com/document/d/1J9fLTCD_j78hCeX-YOq11yZXgpOL-1RTkZV4nQc8v-o/edit?usp=sharing)):*

* Make the students aware of the workshop sheet.  
* Hover around the class, and be available to give hints for processes that may be interesting to consider e.g., generalisations of some of the processes introduced in the lectures.   
* Give time warnings about when the presentation period will begin.

## Learn more {#learn-more}

* Beyond toy architectures – in-context learning   
  * LLMs have an amazing ability to adapt internal representations in-context – \[Read: ICLR: [In-Context Learning of Representations](https://arxiv.org/pdf/2501.00070)\]  
  * LLMs can predict HMM data in-context – \[Read: [Pre-trained Large Language Models Learn Hidden Markov Models In-context](https://arxiv.org/pdf/2506.07298)\]  
  * This suggests that internal representations of LLMs predicting GHMM data in-context, may resemble belief states – \[Read (project proposal, ongoing research): [Context-induced belief geometry in LLMs](https://docs.google.com/document/d/1vqEXWRG7fS3_v2IRwr1PNuyg1KUcCtynxTIdzJ9O6Lo/edit?usp=sharing)\]  
* Processes consisting of GHMMs with richer structure \[Read: [Transformers learn factored representations](https://arxiv.org/pdf/2602.02385v1)\]  
* An explanation of how beliefs are constructed in transformers trained on the Mess3 process \[Read: [Constrained Belief Updates Explain Geometric Structures in Transformer Representations](https://arxiv.org/pdf/2502.01954)\]  
* Review article of computational mechanics \[Read: [Between order and chaos](https://www.nature.com/articles/nphys2190)\]  
* [Simplex](https://www.simplexaisafety.com/) is a non-profit research organisation working on applying computational mechanics to AI safety.

# Teaching guide {#teaching-guide}

## Time Schedule {#time-schedule}

- 10:00–10:30  
  - Lecture: Overview & Scope  
- 10:30–11:00  
  - Lecture: Computational Mechanics Foundations – Hidden Markov Models  
- 11:00–12:00  
  - Exercises: Computational Mechanics Foundations – Hidden Markov Models  
- 12:00–12:30  
  - Lecture: Computational Mechanics Foundations – Prediction  
- (12:30–13:30)  
  - Lunch  
- 13:30–14:30  
  - Exercises: Computational Mechanics Foundations – Prediction  
- 14:30–15:30  
  - Reading Session: Applications – Belief Geometry in Transformers  
- (15:30–16:00)  
  - Afternoon break  
- 16:00–17:45  
  - Workshop: Applications – Designing Processes Worth Studying  
- 17:45–18:00  
  - Feedback form

## Required materials {#required-materials}

* Laptop  
* Projector  
* Whiteboard

## Overview & Scope 

Lecture:

* Deliver the lecture (25 mins), and leave time (5 mins) for questions.

## Computational Mechanics Foundations – Hidden Markov Models

Lecture:

* Deliver the lecture (25 mins), and leave time (5 mins) for questions.

Exercises:

* Give a brief overview of the exercises. Encourage the students to use the hints / solutions if an exercise is taking longer than the assigned time.  
* Direct the students to work freely on the exercises, and be available to answer questions.  
* Give a 10 minute warning before the session ends.

## Computational Mechanics Foundations – Prediction

\[As above\]

## Applications – Belief Geometry in Transformers

Reading session:

* Start by briefly outlining the structure of the reading session: select one of two readings, then discuss in small groups.  
* Direct the students to start the reading, and be available to answer questions.   
* Give a 10 minute warning before the discussion begins.

Workshop:

* Start by briefly outlining the structure of the workshop: design a process, prepare a presentation on your process, present / watch classmates' presentations.   
* Outline the desiderata that one may consider endowing their process with.   
* Direct the students to work freely on the process design, and be available to answer questions & give suggestions if students are stuck.   
* Give the students a 30 minute and a 10 minute warning before the start of the presentations. 

# Postprocessing {#postprocessing}

Lecture content:

* Present both Mealy and Moore HMMs and show that Mealy contains Moore.   
* Update the notation associated with hidden states (always use S’s) and token emissions (always use X’s) when specifying transition matrix elements.  
* Currently HMM minimality is motivated by a fairly trivial example, it would be nice to present a simple example where the minimal HMM is not unique.   
* Include a formal statement about minimality e.g., via a factorisation of the matrix of conditional probabilities.

Exercise/reading/workshop content:

* Not all participants were familiar with python, here are some ways we could accommodate them:  
  * Restructure the exercises to be language agnostic (but have the solutions in python say).   
  * Have a coding track and a pen & paper track.    
* Refine the hints to the exercises so not to give too much away.  
* Include more time for the reading session.  
* Students seemed to find it difficult to design processes from scratch, this session could benefit from starting with some examples and suggestions e.g., random-random modular addition, Mess4.

[image1]: <data:image/png;base64,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>