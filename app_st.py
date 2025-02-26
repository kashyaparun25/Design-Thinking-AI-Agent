import streamlit as st
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# For simplicity, we'll include the classes here for a complete example
class DecisionTracker:
    """Tracks decisions made throughout the design thinking process"""
    
    def __init__(self):
        self.decisions = []
        
    def record_decision(self, stage: str, decision: str, rationale: str):
        """Record a decision with its rationale"""
        self.decisions.append({
            "stage": stage,
            "decision": decision,
            "rationale": rationale,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def get_decision_log(self) -> str:
        """Get a formatted log of all decisions"""
        if not self.decisions:
            return "No decisions recorded."
            
        log = "# Design Thinking Decision Log\n\n"
        for decision in self.decisions:
            log += f"## {decision['stage']} - {decision['timestamp']}\n"
            log += f"**Decision:** {decision['decision']}\n\n"
            log += f"**Rationale:** {decision['rationale']}\n\n"
            log += "---\n\n"
            
        return log

class DesignThinkingCrew:
    """Design Thinking crew for user-centered problem solving with feedback loops"""
    
    def __init__(self, api_keys: Dict[str, str], model_name: str = "gemini/gemini-2.0-flash-thinking-exp-01-21"):
        """Initialize the Design Thinking Crew with necessary API keys and model selection"""
        # Set Serper API key in environment
        os.environ["SERPER_API_KEY"] = api_keys.get("serper")
        
        # Initialize LLM based on selected model
        if "deepseek" in model_name.lower():
            self.llm = LLM(
                model=model_name,  # openrouter/deepseek/deepseek-r1
                base_url="https://openrouter.ai/api/v1",
                api_key=api_keys.get("deepseek")
            )
        elif "gemini" in model_name.lower():
            self.llm = LLM(
                model=model_name,   #gemini/gemini-2.0-flash-thinking-exp-01-21
                api_key=api_keys.get("gemini")
            )
        else:
            # Default to OpenAI
            self.llm = LLM(
                model=model_name,
                api_key=api_keys.get("openai")
            )
        
        # Initialize decision tracker
        self.decision_tracker = DecisionTracker()

        # Initialize tools
        self.search_tool = SerperDevTool(api_key=api_keys.get("serper"))
        
        # Initialize agents with refined definitions
        self.empathize_agent = Agent(
            role="Empathy Researcher",
            goal="Gather deep qualitative insights about user needs and pain points to build a foundation for user-centered design",
            backstory="""You are an expert ethnographic researcher with 15 years of experience in 
            understanding user needs through interviews, surveys, observations, and digital analytics. 
            You excel at identifying underlying pain points, user motivations, and unspoken needs. 
            Your specialty is creating comprehensive empathy maps that bring user perspectives to life 
            and reveal unexpected insights. You're particularly skilled at researching diverse user groups 
            and uncovering cross-cultural nuances that others might miss.""",
            tools=[self.search_tool],
            verbose=True,
            llm=self.llm,

        )
        
        self.define_agent = Agent(
            role="Problem Definition Specialist",
            goal="Synthesize research insights into clear, actionable problem statements that capture the essence of user needs",
            backstory="""You are a master analyst with a unique ability to identify patterns in qualitative data
            and transform them into precise problem definitions. With a background in both data science and
            psychology, you can translate user needs into actionable problem statements that get to the heart
            of the issue. You excel at creating detailed user personas that feel like real people, complete with
            motivations, frustrations, and goals. Your problem definitions have guided numerous successful 
            products because they balance specificity with opportunity for innovation.""",
            tools=[self.search_tool],  # Added internet search tool
            verbose=True,
            llm=self.llm
        )
        
        self.ideate_agent = Agent(
            role="Innovation Facilitator",
            goal="Generate innovative, user-centered solutions through structured creative techniques and lateral thinking",
            backstory="""You are a world-class innovation expert who has mastered the art of ideation.
            Having facilitated hundreds of successful brainstorming sessions across industries, you know
            exactly how to apply techniques like SCAMPER, mind mapping, and analogical thinking to
            generate breakthrough ideas. You're known for your ability to balance wild creativity with
            practical constraints, and for pushing teams beyond obvious solutions to discover truly
            innovative approaches. Your specialty is connecting seemingly unrelated concepts to create
            novel solutions that address user needs in unexpected ways.""",
            tools=[self.search_tool],  # Added internet search tool
            verbose=True,
            llm=self.llm
        )
        
        self.prototype_agent = Agent(
            role="Prototyping Specialist",
            goal="Transform abstract ideas into tangible prototypes that effectively communicate the solution concept",
            backstory="""You are a versatile prototyping expert who knows how to bring ideas to life
            in the most appropriate fidelity for each stage of development. With experience in both
            physical and digital prototyping across multiple industries, you understand exactly what
            level of detail is needed to test key assumptions. You excel at defining core features and
            planning iterative development approaches that maximize learning while minimizing resource
            use. Your prototypes are renowned for clearly communicating the essence of an idea while
            being flexible enough to incorporate feedback.""",
            tools=[self.search_tool],  # Added internet search tool
            verbose=True,
            llm=self.llm
        )
        
        self.test_agent = Agent(
            role="User Testing Coordinator",
            goal="Design and conduct rigorous user testing to validate solutions and gather actionable insights for refinement",
            backstory="""You are a testing methodologist with expertise in both qualitative and
            quantitative user research methods. Having conducted thousands of user tests across
            digital and physical products, you know exactly how to design testing protocols that
            reveal genuine user reactions rather than confirmation bias. You excel at analyzing
            feedback patterns to extract actionable insights, and your testing approaches are known
            for efficiently validating the most critical assumptions first. You have a talent for
            creating testing environments where users feel comfortable providing honest feedback.""",
            tools=[self.search_tool],  # Added internet search tool
            verbose=True,
            llm=self.llm
        )
        
        self.process_manager = Agent(
            role="Design Thinking Process Manager",
            goal="Coordinate communication and manage the workflow across all design thinking stages for optimal collaboration",
            backstory="""You are a seasoned design thinking facilitator who has guided hundreds of
            successful projects from research through implementation. Your expertise lies in ensuring
            the entire process runs smoothly by maintaining clear communication channels between specialists
            at each stage. You excel at synthesizing information across phases, identifying connections
            between insights, and ensuring the team maintains focus on user needs throughout the process.
            Your specialty is knowing when to push forward and when to loop back to earlier stages based
            on new insights, creating a truly iterative and responsive design process.""",
            verbose=True,
            llm=self.llm  # No internet search tool for this agent
        )
        
        self.reporting_agent = Agent(
            role="Design Process Reporter",
            goal="Create comprehensive, clear documentation that captures the entire design thinking journey and its outcomes",
            backstory="""You are an expert technical writer and design documentarian with a gift for
            translating complex design processes into clear, engaging narratives. With experience
            documenting design projects across industries, you know exactly how to structure reports
            that highlight key insights, decisions, and outcomes at each stage. You excel at creating
            visually appealing documentation that balances detail with readability, ensuring that 
            stakeholders at all levels can understand the process and its value. Your specialty is
            crafting documentation that serves both as a record of the project and as a resource
            for future design initiatives.""",
            verbose=True,
            llm=self.llm  # No internet search tool for this agent
        )

    def generate_challenge(self, domain: str, context: str = None, constraints: List[str] = None) -> Dict[str, Any]:
        """Let the LLM define a design challenge based on the provided domain, context, and constraints
        
        Args:
            domain: The general field or area (e.g., "Educational Technology", "Healthcare")
            context: Background information about the problem space and users
            constraints: List of specific requirements or limitations
            
        Returns:
            Dictionary containing the challenge statement, context, and constraints
        """
        # Set defaults if not provided
        context = context or "No specific context provided."
        constraints = constraints or ["No specific constraints provided."]
        
        challenge_agent = Agent(
            role="Design Challenge Formulator",
            goal="Define a compelling design challenge based on domain, context, and constraints",
            backstory="""You are an expert at framing meaningful design challenges.
            You excel at identifying problems worth solving by analyzing contexts and constraints.
            Your specialty is crafting "How might we" statements that are specific enough to
            guide work but open enough to allow for innovation. You're known for your ability to
            identify the core user needs hidden within complex problem spaces.""",
            tools=[self.search_tool],  # Add search tool to research the domain if needed
            verbose=True,
            llm=self.llm
        )
        
        challenge_task = Task(
            description=f"""Analyze the given domain, context, and constraints to generate a well-defined design challenge.
            
            DOMAIN: {domain}
            
            CONTEXT: {context}
            
            CONSTRAINTS:
            {chr(10).join(['- ' + constraint for constraint in constraints])}
            
            First, research the domain to identify key trends, problems, and opportunities.
            Then, analyze the context to understand stakeholders and their needs.
            Finally, consider the constraints to ensure the challenge is feasible.
            
            The output should be a comprehensive design challenge definition.
            """,
            expected_output="""A complete design challenge definition formatted as follows:
            
            # CHALLENGE STATEMENT
            [A clear "How might we..." statement that defines the problem to solve]
            
            # CONTEXT
            [3-5 paragraphs explaining why this challenge matters, including:
            - Current situation analysis
            - User pain points and needs
            - Relevant trends or data
            - Stakeholder information]
            
            # CONSTRAINTS
            [A comprehensive list of constraints that solutions must address, including:
            - Technical requirements
            - User needs
            - Business or organizational limitations
            - Ethical considerations]
            
            # SUCCESS CRITERIA
            [Measurable criteria that will determine if solutions to this challenge are successful]
            """,
            agent=challenge_agent
        )
        
        # Create a temporary crew just to generate the challenge
        challenge_crew = Crew(
            agents=[challenge_agent],
            tasks=[challenge_task],
            verbose=True
        )
        
        result = challenge_crew.kickoff(inputs={"domain": domain})
        
        # Parse the result into structured components
        try:
            # Split by headers
            sections = result.raw.split('# ')
            
            # Initialize dictionary with default values
            challenge_data = {
                "challenge": "No challenge statement generated",
                "context": "No context generated",
                "constraints": "No constraints generated",
                "success_criteria": "No success criteria generated"
            }
            
            # Process each section
            for section in sections:
                if not section.strip():
                    continue
                    
                # Split the header from content
                parts = section.split('\n', 1)
                if len(parts) < 2:
                    continue
                    
                header, content = parts[0].strip(), parts[1].strip()
                
                if "CHALLENGE STATEMENT" in header.upper():
                    challenge_data["challenge"] = content
                elif "CONTEXT" in header.upper():
                    challenge_data["context"] = content
                elif "CONSTRAINTS" in header.upper():
                    challenge_data["constraints"] = content
                elif "SUCCESS CRITERIA" in header.upper():
                    challenge_data["success_criteria"] = content
        
        except Exception as e:
            print(f"Error parsing challenge output: {e}")
            # Fallback to simple parsing
            challenge_data = {
                "challenge": result.raw.split('\n', 1)[0] if '\n' in result.raw else result.raw,
                "context": result.raw,
                "constraints": ", ".join(constraints)
            }
        
        # Record this as a decision
        self.decision_tracker.record_decision(
            stage="Challenge Definition",
            decision=f"Defined challenge: {challenge_data['challenge']}",
            rationale="Generated by AI based on domain, context, and constraints analysis"
        )
        
        return challenge_data
        
    def get_all_agents(self):
        """Return all agents for display in the UI"""
        return {
            "Empathy Researcher": self.empathize_agent,
            "Problem Definition Specialist": self.define_agent,
            "Innovation Facilitator": self.ideate_agent,
            "Prototyping Specialist": self.prototype_agent,
            "User Testing Coordinator": self.test_agent,
            "Design Thinking Process Manager": self.process_manager,
            "Design Process Reporter": self.reporting_agent
        }
        
    def run_task(self, task_name, task, project_input, context_tasks=None):
        """Run a single task and return its result"""
        if context_tasks:
            task.context = context_tasks
            
        # Create a temporary crew for this single task
        temp_crew = Crew(
            agents=[task.agent],
            tasks=[task],
            verbose=True
        )
        
        # Add extensive error handling and validation
        try:
            result = temp_crew.kickoff(inputs=project_input)
            
            # Validate result
            if not result:
                st.error("Task execution returned no result")
                # Create a fallback result
                from crewai.tasks.task_output import TaskOutput
                result = TaskOutput(
                    task_id=task_name,
                    raw="Task execution failed - please try again",
                    agent=task.agent.role
                )
                
            # Ensure result has required attributes
            if not hasattr(result, 'raw'):
                st.error("Task result missing required attributes")
                result.raw = "Task execution produced invalid output format - please try again"
                
            return result
            
        except Exception as e:
            st.error(f"Error during task execution: {str(e)}")
            # Create error result
            from crewai.tasks.task_output import TaskOutput
            return TaskOutput(
                task_id=task_name,
                raw=f"Task execution failed with error: {str(e)}",
                agent=task.agent.role
            )

# Define task definitions globally so it can be accessed by multiple functions
def get_task_definitions(session_state):
    """Get task definitions for the design thinking process"""
    if not session_state.crew or not session_state.project_input:
        return {}
    
    return {
        "empathize": {
            "name": "Empathize",
            "description": f"""Conduct user research for: {session_state.project_input['challenge']}

            CHALLENGE: 
            {session_state.project_input.get('challenge', 'No challenge provided.')}

            CONTEXT:
            {session_state.project_input.get('context', 'No context provided.')}
            
            CONSTRAINTS:
            {session_state.project_input.get('constraints', 'No specific constraints.')}

            Gather insights through analyzing user behavior, needs, and pain points.
            Consider all stakeholders involved.""",
            "expected_output": """Detailed empathy map with:
            1. User observations
            2. Identified pain points
            3. Key user needs
            4. Stakeholder insights""",
            "agent": session_state.crew.empathize_agent,
            "human_input": True
        },
        "define": {
            "name": "Define",
            "description": """Based on the research insights, define the core problem.
            
            Create user personas and establish clear success metrics.""",
            "expected_output": """1. Clear problem statement
            2. Primary user persona
            3. Key success metrics
            4. List of user requirements""",
            "agent": session_state.crew.define_agent,
            "human_input": True
        },
        "ideate": {
            "name": "Ideate",
            "description": """Generate innovative solutions for the defined problem.
            Use creative techniques to explore multiple approaches.""",
            "expected_output": """1. List of potential solutions
            2. Evaluation of each idea
            3. Prioritized concepts
            4. Innovation opportunities""",
            "agent": session_state.crew.ideate_agent,
            "human_input": True
        },
        "prototype": {
            "name": "Prototype",
            "description": """Create a prototype plan for the top solution(s).
            Define key features and development milestones.""",
            "expected_output": """1. Prototype specifications
            2. Core features list
            3. Development milestones
            4. Resource requirements""",
            "agent": session_state.crew.prototype_agent,
            "human_input": True
        },
        "test": {
            "name": "Test",
            "description": """Design a testing protocol for the prototype.
            Include methods for gathering and analyzing user feedback.""",
            "expected_output": """1. Testing protocol
            2. Feedback collection methods
            3. Success criteria
            4. Iteration recommendations""",
            "agent": session_state.crew.test_agent,
            "human_input": True
        },
        "process": {
            "name": "Process Management",
            "description": """Monitor and coordinate the entire design thinking process.
            Ensure information flows effectively between stages and maintain alignment
            with project objectives. Identify connections between insights from different
            stages and highlight opportunities for iteration.""",
            "expected_output": """1. Process coordination notes
            2. Cross-stage insights and connections
            3. Workflow optimization recommendations
            4. Progress tracking summary""",
            "agent": session_state.crew.process_manager
        },
        "decisions": {
            "name": "Decision Documentation",
            "description": """Document all key decisions made during the design thinking process,
            including the rationale behind each decision.""",
            "expected_output": """A comprehensive decision log including:
            1. Decisions made at each stage
            2. Rationale for each decision
            3. Alternatives considered
            4. Impact on the overall design process""",
            "agent": session_state.crew.reporting_agent
        },
        "report": {
            "name": "Final Report",
            "description": """Create a comprehensive markdown report documenting the entire
            design thinking process. Include insights and outcomes from each stage.""",
            "expected_output": """A detailed markdown report including:
            1. Executive Summary
            2. User Research Findings
            3. Problem Definition
            4. Ideation Process and Solutions
            5. Prototype Details
            6. Testing Results and Recommendations
            7. Next Steps""",
            "agent": session_state.crew.reporting_agent
        }
    }

# Initialize session state
def init_session_state():
    if 'project_input' not in st.session_state:
        st.session_state.project_input = None
    
    if 'completed_tasks' not in st.session_state:
        st.session_state.completed_tasks = {}
    
    if 'human_feedback' not in st.session_state:
        st.session_state.human_feedback = {}
    
    if 'stage_suggestions' not in st.session_state:
        st.session_state.stage_suggestions = {}
        
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = None
        
    if 'task_history' not in st.session_state:
        st.session_state.task_history = []
        
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            "gemini": "",
            "serper": "",
            "openai": "",
            "deepseek": ""
        }
        
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "gemini/gemini-2.0-flash-thinking-exp-01-21"
        
    if 'crew' not in st.session_state:
        st.session_state.crew = None
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

# Functions for the Streamlit UI
def setup_api_keys():
    st.sidebar.header("API Keys")
    
    gemini_key = st.sidebar.text_input("Gemini API Key", value=st.session_state.api_keys.get("gemini", ""), type="password")
    serper_key = st.sidebar.text_input("Serper API Key", value=st.session_state.api_keys.get("serper", ""), type="password")
    openai_key = st.sidebar.text_input("OpenAI API Key (Optional)", value=st.session_state.api_keys.get("openai", ""), type="password")
    deepseek_key = st.sidebar.text_input("DeepSeek API Key (Optional)", value=st.session_state.api_keys.get("deepseek", ""), type="password")
    
    st.session_state.api_keys = {
        "gemini": gemini_key,
        "serper": serper_key,
        "openai": openai_key,
        "deepseek": deepseek_key
    }
    
    # Model selection
    model_options = {
        "Gemini 2.0 Flash Thinking": "gemini/gemini-2.0-flash-thinking-exp-01-21",
        "DeepSeek Reasoner": "openrouter/deepseek/deepseek-r1",
        "OpenAI GPT-4": "gpt-4"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select LLM Model",
        options=list(model_options.keys()),
        index=0
    )
    
    st.session_state.model_name = model_options[selected_model]
    
    # Initialize crew if keys are provided
    if gemini_key and serper_key:
        if st.session_state.crew is None or st.sidebar.button("Reinitialize Crew"):
            try:
                #langtrace.init(api_key='3c69b64050b8794bcea55157a2a479a1a3037a1cf57b7bef18c7fbde6cb58e62')
                st.session_state.crew = DesignThinkingCrew(api_keys=st.session_state.api_keys, model_name=st.session_state.model_name)
                st.sidebar.success("Design Thinking Crew initialized successfully!")
            except Exception as e:
                st.sidebar.error(f"Error initializing crew: {e}")
    else:
        st.sidebar.warning("Please provide Gemini and Serper API keys to initialize the crew.")

def setup_challenge():
    st.header("Design Challenge Setup")
    
    challenge_method = st.radio(
        "How would you like to define the design challenge?",
        ["Provide challenge details", "Generate from domain", "Extract from context"],
        horizontal=True
    )
    
    if challenge_method == "Provide challenge details":
        challenge = st.text_area("Challenge Statement", placeholder="Enter the design challenge statement...")
        context = st.text_area("Context", placeholder="Provide background information about the problem...")
        constraints = st.text_area("Constraints", placeholder="List the constraints, one per line...")
        
        if st.button("Set Challenge"):
            constraints_list = [c.strip() for c in constraints.split("\n") if c.strip()]
            st.session_state.project_input = {
                "challenge": challenge,
                "context": context,
                "constraints": constraints_list if constraints_list else "No specific constraints provided."
            }
            st.success("Challenge set successfully!")
            
    elif challenge_method == "Generate from domain":
        domain = st.text_input("Domain", placeholder="e.g., Educational Technology, Healthcare...")
        context = st.text_area("Context (Optional)", placeholder="Additional background information...")
        constraints = st.text_area("Constraints (Optional)", placeholder="List the constraints, one per line...")
        
        if st.button("Generate Challenge"):
            if st.session_state.crew is None:
                st.error("Please initialize the crew first by providing API keys in the sidebar.")
            else:
                with st.spinner("Generating challenge from domain..."):
                    constraints_list = [c.strip() for c in constraints.split("\n") if c.strip()]
                    try:
                        st.session_state.project_input = st.session_state.crew.generate_challenge(
                            domain=domain,
                            context=context if context else None,
                            constraints=constraints_list if constraints_list else None
                        )
                        st.success("Challenge generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating challenge: {e}")
                        
    elif challenge_method == "Extract from context":
        context = st.text_area("Context", placeholder="Describe the situation in detail...")
        constraints = st.text_area("Constraints (Optional)", placeholder="List the constraints, one per line...")
        
        if st.button("Extract Challenge"):
            if st.session_state.crew is None:
                st.error("Please initialize the crew first by providing API keys in the sidebar.")
            else:
                with st.spinner("Analyzing context and extracting challenge..."):
                    constraints_list = [c.strip() for c in constraints.split("\n") if c.strip()]
                    try:
                        # Create a context analysis agent and analyze the context
                        context_analysis_agent = Agent(
                            role="Context Analyst",
                            goal="Analyze context to identify key challenges and opportunities",
                            backstory="You excel at reading between the lines to identify core problems that need addressing.",
                            tools=[st.session_state.crew.search_tool],
                            verbose=True,
                            llm=st.session_state.crew.llm
                        )
                        
                        context_analysis_task = Task(
                            description=f"Analyze the following context to identify challenges: \n\n{context}",
                            expected_output="Analysis with domain identification and challenge statement",
                            agent=context_analysis_agent
                        )
                        
                        analysis_crew = Crew(
                            agents=[context_analysis_agent],
                            tasks=[context_analysis_task],
                            verbose=True
                        )
                        
                        analysis_result = analysis_crew.kickoff()
                        
                        # Extract domain from analysis
                        try:
                            domain_section = analysis_result.raw.split('# DOMAIN IDENTIFICATION')[1].split('#')[0].strip()
                        except:
                            domain_section = "General Design"
                            
                        # Generate full challenge
                        st.session_state.project_input = st.session_state.crew.generate_challenge(
                            domain=domain_section,
                            context=context,
                            constraints=constraints_list if constraints_list else None
                        )
                        st.success("Challenge extracted successfully!")
                    except Exception as e:
                        st.error(f"Error extracting challenge: {e}")

def display_challenge():
    if st.session_state.project_input:
        st.subheader("Current Design Challenge")
        
        with st.expander("Challenge Details", expanded=True):
            st.markdown(f"### 🎯 Challenge Statement")
            st.markdown(f"{st.session_state.project_input.get('challenge', 'No challenge statement')}")
            
            st.markdown(f"### 📚 Context")
            st.markdown(f"{st.session_state.project_input.get('context', 'No context provided')}")
            
            st.markdown(f"### 🧩 Constraints")
            constraints = st.session_state.project_input.get('constraints', [])
            if isinstance(constraints, list):
                for constraint in constraints:
                    st.markdown(f"- {constraint}")
            else:
                st.markdown(constraints)
                
            if 'success_criteria' in st.session_state.project_input:
                st.markdown(f"### 🏆 Success Criteria")
                st.markdown(f"{st.session_state.project_input.get('success_criteria', 'No success criteria defined')}")

def run_design_thinking_process():
    if not st.session_state.project_input:
        st.warning("Please set up a design challenge first.")
        return
        
    if st.session_state.crew is None:
        st.error("Please initialize the crew first by providing API keys in the sidebar.")
        return
        
    # Get the task definitions
    task_definitions = get_task_definitions(st.session_state)
    
    # Display navigation options
    st.subheader("Design Thinking Process")
    
    task_order = list(task_definitions.keys())
    
    # If no current stage is set, use the first one
    if st.session_state.current_stage is None:
        st.session_state.current_stage = task_order[0]
        
    # Show tabs for all stages
    tabs = st.tabs([task_definitions[stage]["name"] for stage in task_order])
    current_index = task_order.index(st.session_state.current_stage)
    
    # Apply any existing suggestions to the task description
    current_task_def = task_definitions[st.session_state.current_stage]
    task_description = current_task_def["description"]
    
    if st.session_state.current_stage in st.session_state.stage_suggestions:
        task_description += f"\n\nHUMAN SUGGESTIONS TO INCORPORATE:\n{st.session_state.stage_suggestions[st.session_state.current_stage]}"
    
    # Create the task
    task = Task(
        description=task_description,
        expected_output=current_task_def["expected_output"],
        agent=current_task_def["agent"]
    )
    
    # Set context based on completed tasks
    if st.session_state.current_stage != "empathize":
        context_tasks = []
        
        # Define context based on design thinking flow
        if st.session_state.current_stage == "define" and "empathize" in st.session_state.completed_tasks:
            context_tasks.append(st.session_state.completed_tasks["empathize"])
            
        elif st.session_state.current_stage == "ideate":
            if "define" in st.session_state.completed_tasks:
                context_tasks.append(st.session_state.completed_tasks["define"])
            if "empathize" in st.session_state.completed_tasks:
                context_tasks.append(st.session_state.completed_tasks["empathize"])
            
        elif st.session_state.current_stage == "prototype":
            if "ideate" in st.session_state.completed_tasks:
                context_tasks.append(st.session_state.completed_tasks["ideate"])
            if "define" in st.session_state.completed_tasks:
                context_tasks.append(st.session_state.completed_tasks["define"])
        
        elif st.session_state.current_stage == "test":
            if "prototype" in st.session_state.completed_tasks:
                context_tasks.append(st.session_state.completed_tasks["prototype"])
            if "ideate" in st.session_state.completed_tasks:
                context_tasks.append(st.session_state.completed_tasks["ideate"])
            
        elif st.session_state.current_stage == "process":
            for name in ["empathize", "define", "ideate", "prototype", "test"]:
                if name in st.session_state.completed_tasks:
                    context_tasks.append(st.session_state.completed_tasks[name])
            
        elif st.session_state.current_stage == "decisions":
            if "process" in st.session_state.completed_tasks:
                context_tasks.append(st.session_state.completed_tasks["process"])
            
        elif st.session_state.current_stage == "report":
            for name, task_obj in st.session_state.completed_tasks.items():
                context_tasks.append(task_obj)
                
        # Set the context
        if context_tasks:
            task.context = context_tasks
    
    # Display the current active tab
    with tabs[current_index]:
        st.write(f"**Agent:** {current_task_def['agent'].role}")
        
        # Display the task output (if completed)
        if st.session_state.current_stage in st.session_state.completed_tasks:
            st.success("Task completed!")
            with st.expander("Task Output", expanded=True):
                st.markdown(st.session_state.completed_tasks[st.session_state.current_stage].output.raw)
                
            # Display any previous feedback
            if st.session_state.current_stage in st.session_state.human_feedback:
                with st.expander("Previous Feedback"):
                    for i, feedback in enumerate(st.session_state.human_feedback[st.session_state.current_stage]):
                        st.markdown(f"**Feedback {i+1}:** {feedback}")
            
            # Option to revise the task
            if st.button("Revise this task"):
                st.session_state.completed_tasks.pop(st.session_state.current_stage, None)
                st.rerun()
                
        else:
            # Display the task description and expected output
            st.markdown("### Task Description")
            st.markdown(task_description)
            
            st.markdown("### Expected Output")
            st.markdown(current_task_def["expected_output"])
            
            # Button to run the task
            if st.button("Run Task"):
                with st.spinner(f"Running {current_task_def['name']} task..."):
                    try:
                        result = st.session_state.crew.run_task(
                            task_name=st.session_state.current_stage,
                            task=task,
                            project_input=st.session_state.project_input,
                            context_tasks=task.context if hasattr(task, 'context') else None
                        )
                        
                        # Store the completed task
                        st.session_state.completed_tasks[st.session_state.current_stage] = task
                        
                        # Add task to history
                        st.session_state.task_history.append({
                            "stage": st.session_state.current_stage,
                            "agent": current_task_def['agent'].role,
                            "output": result.raw
                        })
                        
                        # Add a message to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**{current_task_def['agent'].role}**: I've completed the {current_task_def['name']} task!",
                            "avatar": "🧠"
                        })
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error executing task: {e}")
            
        # Add direct feedback section within each stage tab
        if current_task_def.get("human_input", False):
            st.markdown("### Provide Feedback")
            stage_feedback = st.text_area("Your suggestions for this stage:", key=f"feedback_{st.session_state.current_stage}")
            if st.button("Submit Feedback", key=f"submit_{st.session_state.current_stage}"):
                # Store the feedback
                if st.session_state.current_stage not in st.session_state.human_feedback:
                    st.session_state.human_feedback[st.session_state.current_stage] = []
                st.session_state.human_feedback[st.session_state.current_stage].append(stage_feedback)
                
                # Force task re-execution with feedback
                st.session_state.completed_tasks.pop(st.session_state.current_stage, None)
                
                # Add feedback to task description
                task_description += f"\n\nPLEASE INCORPORATE THIS FEEDBACK:\n{stage_feedback}"
                task = Task(
                    description=task_description,
                    expected_output=current_task_def["expected_output"],
                    agent=current_task_def["agent"]
                )
                
                # Re-run the task immediately
                with st.spinner(f"Re-running task with feedback..."):
                    result = st.session_state.crew.run_task(
                        task_name=st.session_state.current_stage,
                        task=task,
                        project_input=st.session_state.project_input
                    )
                
                # Record the decision
                st.session_state.crew.decision_tracker.record_decision(
                    stage=f"{task_definitions[st.session_state.current_stage]['name']} Feedback",
                    decision=f"Received direct stage feedback",
                    rationale=f"Human provided: {stage_feedback}"
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"[Feedback for {task_definitions[st.session_state.current_stage]['name']}]: {stage_feedback}"
                })
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"**{current_task_def['agent'].role}**: Thank you for your feedback! I'll incorporate this into my work.",
                    "avatar": "🧠"
                })
                
                st.success("Feedback submitted successfully!")
                st.rerun()
    
    # Navigation controls
    st.subheader("Navigation")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Previous button
        if current_index > 0:
            if st.button("⬅️ Previous Stage"):
                st.session_state.current_stage = task_order[current_index - 1]
                st.rerun()
    
    with col3:
        # Next button
        if current_index < len(task_order) - 1:
            if st.button("Next Stage ➡️"):
                st.session_state.current_stage = task_order[current_index + 1]
                st.rerun()
    
    with col2:
        # Jump to any stage
        jump_options = ["Jump to..."] + [task_definitions[stage]["name"] for stage in task_order]
        jump_to = st.selectbox("", jump_options, index=0)
        
        if jump_to != "Jump to...":
            target_stage = task_order[[task_definitions[stage]["name"] for stage in task_order].index(jump_to)]
            if st.button(f"Go to {jump_to}"):
                st.session_state.current_stage = target_stage
                st.rerun()

def display_chat():
    st.subheader("Human-AI Collaboration")
    
    # Get task definitions
    task_definitions = get_task_definitions(st.session_state)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar=message.get("avatar", "🧠")):
                st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Provide feedback or suggestions...", key="chat_input")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Process the input
        if st.session_state.current_stage and st.session_state.current_stage in task_definitions:
            # Record feedback for the current stage
            if st.session_state.current_stage not in st.session_state.human_feedback:
                st.session_state.human_feedback[st.session_state.current_stage] = []
                
            st.session_state.human_feedback[st.session_state.current_stage].append(user_input)
            
            # Record the decision
            if st.session_state.crew:
                st.session_state.crew.decision_tracker.record_decision(
                    stage=f"{task_definitions[st.session_state.current_stage]['name']} Feedback",
                    decision=f"Received human feedback",
                    rationale=f"Human provided: {user_input}"
                )
            
            # Add agent response
            current_agent_role = task_definitions[st.session_state.current_stage]["agent"].role
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**{current_agent_role}**: Thank you for your feedback! I'll incorporate this into my work.",
                "avatar": "🧠"
            })
            
            # Ask if this should be applied as a suggestion for future stages
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Would you like to apply this as a suggestion for any specific stage? Type 'suggest for [stage name]' or 'no'.",
                "avatar": "🤔"
            })
        
        st.rerun()
    
    # Process suggestion requests
    if len(st.session_state.chat_history) >= 2 and st.session_state.chat_history[-2]["role"] == "assistant" and "Would you like to apply this as a suggestion" in st.session_state.chat_history[-2]["content"]:
        last_user_message = next((msg for msg in reversed(st.session_state.chat_history) if msg["role"] == "user"), None)
        
        if last_user_message and "suggest for" in last_user_message["content"].lower():
            # Extract stage name
            stage_text = last_user_message["content"].lower().split("suggest for ")[1].strip()
            target_stage = next((stage for stage in task_definitions.keys() if stage in stage_text or task_definitions[stage]["name"].lower() in stage_text), None)
            
            if target_stage:
                # Get the feedback message (second to last user message)
                feedback_messages = [msg for msg in st.session_state.chat_history if msg["role"] == "user"]
                if len(feedback_messages) >= 2:
                    feedback = feedback_messages[-2]["content"]
                    
                    # Add suggestion
                    if target_stage in st.session_state.stage_suggestions:
                        st.session_state.stage_suggestions[target_stage] += f"\n- {feedback}"
                    else:
                        st.session_state.stage_suggestions[target_stage] = f"- {feedback}"
                    
                    # Add confirmation message
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Suggestion added for the {task_definitions[target_stage]['name']} stage!",
                        "avatar": "✅"
                    })
                    
                    # Record the decision
                    if st.session_state.crew:
                        st.session_state.crew.decision_tracker.record_decision(
                            stage="Suggestion",
                            decision=f"Added suggestion for {task_definitions[target_stage]['name']} stage",
                            rationale=f"Human suggested: {feedback}"
                        )
                    
                    st.rerun()
            elif last_user_message["content"].lower() == "no":
                # Add acknowledgment message
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "No problem! Your feedback has been recorded for the current stage only.",
                    "avatar": "👍"
                })
                st.rerun()

def display_decision_log():
    if st.session_state.crew:
        with st.expander("Decision Log"):
            decision_log = st.session_state.crew.decision_tracker.get_decision_log()
            st.markdown(decision_log)

# Main Streamlit UI
def main():
    st.set_page_config(
        page_title="Design Thinking AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Setup sidebar with API keys
    setup_api_keys()
    
    # App title
    st.title("🧠 Design Thinking AI")
    st.markdown("Collaborative AI-powered design thinking process")
    
    # Main content area
    if st.session_state.crew is None:
        st.info("Please set up your API keys in the sidebar to get started.")
    else:
        # Tabs for different sections
        main_tabs = st.tabs(["Challenge Setup", "Design Thinking Process", "Chat & Feedback"])
        
        with main_tabs[0]:
            setup_challenge()
            display_challenge()
            
        with main_tabs[1]:
            if st.session_state.project_input:
                run_design_thinking_process()
            else:
                st.info("Please set up a design challenge first in the Challenge Setup tab.")
                
        with main_tabs[2]:
            display_chat()
            display_decision_log()

if __name__ == "__main__":
    main()
