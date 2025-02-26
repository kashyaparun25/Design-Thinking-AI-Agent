# Design Thinking AI Agent 🚀

Welcome to **Design Thinking AI**, an innovative, AI-powered application that reimagines the [design thinking process](https://www.interaction-design.org/literature/topics/design-thinking) through specialized **AI agents**. Built with [Streamlit](https://streamlit.io/), this interactive tool guides users—individuals or teams—through a structured, human-centered approach to problem-solving. By blending the creativity and empathy of design thinking with AI’s efficiency and intelligence, this project empowers you to address complex challenges and collaboratively craft impactful solutions.

---

## Table of Contents

- [Overview](#overview)
- [Workflow Diagram](#workflow-diagram)
- [Detailed Workflow Explanation](#detailed-workflow-explanation)
  - [Challenge Setup](#challenge-setup)
  - [AI Agents and Their Roles](#ai-agents-and-their-roles)
  - [Task Execution and Context Preservation](#task-execution-and-context-preservation)
  - [Human Feedback and Collaboration](#human-feedback-and-collaboration)
  - [Final Report and Decision Logging](#final-report-and-decision-logging)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Summary of Workflow and Agent Roles](#summary-of-workflow-and-agent-roles)

---

## Overview

**Design Thinking AI** is a collaborative ecosystem where human creativity meets artificial intelligence. Powered by a team of specialized **AI agents**, this application guides users through the entire [design thinking process](https://en.wikipedia.org/wiki/Design_thinking)—a methodology centered on solving problems by deeply understanding users’ needs. The classic stages—**Empathize**, **Define**, **Ideate**, **Prototype**, and **Test**—are augmented with additional steps like **Process Management**, **Decision Documentation**, and **Final Reporting**, all driven by AI.

Each **AI agent** plays a pivotal role:
- The **Empathy Researcher** uncovers user pain points with precision.
- The **Innovation Facilitator** sparks creative breakthroughs.
- The **Reporting Agent** delivers a comprehensive final report.

Built with [Streamlit](https://streamlit.io/), the app provides an intuitive interface that fosters human-AI collaboration, capturing feedback at every stage to refine outcomes iteratively. Whether defining a challenge or testing a prototype, the **AI agents** ensure accuracy, creativity, and continuity throughout.

---

## Workflow Diagram

Below is a visual representation of the Design Thinking AI workflow:

```mermaid
flowchart TD
    A[Challenge Setup] --> B[Empathize Stage]
    B --> C[Define Stage]
    C --> D[Ideate Stage]
    D --> E[Prototype Stage]
    E --> F[Test Stage]
    F --> G[Process Management]
    G --> H[Decision Documentation]
    H --> I[Final Report]

    subgraph Agents
      B1[Empathy Researcher]
      C1[Problem Definition Specialist]
      D1[Innovation Facilitator]
      E1[Prototyping Specialist]
      F1[User Testing Coordinator]
      G1[Process Manager]
      H1[Reporting Agent]
    end

    B --> B1
    C --> C1
    D --> D1
    E --> E1
    F --> F1
    G --> G1
    H --> H1
```
---
*Note: You can view or edit this diagram using an online Mermaid editor.*

---

## Detailed Workflow Explanation

### Challenge Setup 🎯

**Purpose:**  
Users define or generate the design challenge—specifying context, constraints, and success criteria—which is stored in `st.session_state.project_input`. This forms the basis for all subsequent tasks.

**Output:**  
A clear, actionable challenge statement that guides the outputs of the AI agents.

---

### AI Agents and Their Roles 🤖

The system is powered by a **Design Thinking Crew** consisting of several specialized AI agents:

- **Empathy Researcher**  
  **Stage:** Empathize  
  **Role:** Gathers deep insights about user behavior and pain points.  
  **Impact:** Establishes the foundational qualitative insights for the process.

- **Problem Definition Specialist**  
  **Stage:** Define  
  **Role:** Synthesizes research data into a clear problem statement and creates user personas.  
  **Impact:** Sharpens focus by clarifying the core problem to address.

- **Innovation Facilitator**  
  **Stage:** Ideate  
  **Role:** Generates creative solutions using techniques like brainstorming and lateral thinking.  
  **Impact:** Expands the range of possibilities with innovative ideas.

- **Prototyping Specialist**  
  **Stage:** Prototype  
  **Role:** Converts ideas into tangible prototype plans, specifying key features and milestones.  
  **Impact:** Bridges abstract ideas to practical, testable models.

- **User Testing Coordinator**  
  **Stage:** Test  
  **Role:** Designs testing protocols and gathers user feedback.  
  **Impact:** Validates the solution and suggests refinements.

- **Design Thinking Process Manager**  
  **Stage:** Process Management  
  **Role:** Oversees the workflow and ensures context is preserved across stages.  
  **Impact:** Maintains continuity throughout the process.

- **Design Process Reporter**  
  **Stage:** Decision Documentation & Final Report  
  **Role:** Documents the complete design journey, compiling insights, outputs, and decisions into a final report.  
  **Impact:** Provides transparency and a historical record of the process.

---

## Installation 🛠️

### Prerequisites

- Python 3.7 or later
- pip (Python package installer)

### Dependencies

Create a `requirements.txt` file with the following content:

```
streamlit
crewai
crewai_tools
```

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/kashyaparun25/Design-Thinking-AI-Agent.git
cd design-thinking-ai
```

---

## Usage 🚀

Run the application with:

```bash
streamlit run main.py
```

---

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Summary of Workflow and Agent Roles

- **Challenge Setup:** Define the design challenge.
- **Empathize:** The **Empathy Researcher** gathers insights.
- **Define:** The **Problem Definition Specialist** creates a problem statement.
- **Ideate:** The **Innovation Facilitator** generates solutions.
- **Prototype:** The **Prototyping Specialist** develops prototypes.
- **Test:** The **User Testing Coordinator** conducts tests.
- **Process Management:** The **Process Manager** maintains workflow.
- **Decision Documentation:** The **Reporting Agent** compiles insights.
- **Final Report:** The **Design Process Reporter** provides a final overview.

---

Enjoy exploring innovative problem-solving with **Design Thinking AI**! 🚀
