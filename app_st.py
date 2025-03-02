__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import base64
import tempfile
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
            allow_delegation=True
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
            llm=self.llm,
            allow_delegation=True
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
            llm=self.llm,
            allow_delegation=True
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
            llm=self.llm,
            allow_delegation=True
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
            llm=self.llm,
            allow_delegation=True
        )
        
        # Create the manager agent
        self.manager_agent = Agent(
            role="Design Thinking Process Manager",
            goal="Coordinate communication and manage the workflow across all design thinking stages for optimal collaboration,  ensuring all work relates to the specific design challenge and constraints provided",
            backstory="""You are a seasoned design thinking facilitator who has guided hundreds of
            successful projects from research through implementation. YOUR MOST IMPORTANT RESPONSIBILITY
            is to ensure all agents focus on the specific design challenge, context, and constraints
            that were provided for this project. You must start by understanding these project parameters and
            ensure that all research, ideation, and other work stays relevant to them. Your expertise lies in ensuring
            the entire process runs smoothly by maintaining clear communication channels between specialists
            at each stage. You excel at synthesizing information across phases, identifying connections
            between insights, and ensuring the team maintains focus on user needs throughout the process.
            Your specialty is knowing when to push forward and when to loop back to earlier stages based
            on new insights, creating a truly iterative and responsive design process.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=True  # Manager can delegate tasks
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
            llm=self.llm,
            allow_delegation=True
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
            "Design Thinking Process Manager": self.manager_agent,
            "Design Process Reporter": self.reporting_agent
        }
        
    def run_task(self, task_name, task, project_input, context_tasks=None, pdf_contents=None):
        """Run a single task and return its result"""
        if context_tasks:
            task.context = context_tasks
            
        # Incorporate PDF contents into task description if available
        if pdf_contents and len(pdf_contents) > 0:
            pdf_context = "\n\nREFERENCE PDF DOCUMENTS:\n"
            for i, content in enumerate(pdf_contents):
                # Limit content length to avoid token issues
                truncated_content = content[:10000] + "..." if len(content) > 10000 else content
                pdf_context += f"\nPDF DOCUMENT {i+1}:\n{truncated_content}\n\n"
            
            task.description += pdf_context
        
        # Make sure the tool is properly initialized
        search_tool = SerperDevTool(api_key=os.environ.get("SERPER_API_KEY", ""))
        
        # Re-initialize the agent with the tool to ensure it's using it properly
        agent_copy = Agent(
            role=task.agent.role,
            goal=task.agent.goal,
            backstory=task.agent.backstory,
            verbose=True,
            tools=[search_tool],  # Explicitly provide the tool
            llm=self.llm
        )
        
        # Create a temporary crew for this single task
        temp_crew = Crew(
            agents=[agent_copy],  # Use the copy with explicit tool
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
                    agent=task.agent.role,
                    description=task.description
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
                agent=task.agent.role,
                description=task.description
            )
# Helper functions for file handling
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return "Error extracting text from PDF"


# Define task definitions globally so it can be accessed by multiple functions
def get_task_definitions(session_state):
    """Get task definitions for the design thinking process"""
    if not session_state.crew or not session_state.project_input:
        return {}
    
    return {
        "empathize": {
            "name": "Empathize",
            "description": f"""Conduct user research for: {session_state.project_input['challenge']}



            Gather insights through analyzing user behavior, needs, and pain points.
            Consider all stakeholders involved.""",
            "expected_output": """Detailed empathy map with:
            1. User observations
            2. Identified pain points
            3. Key user needs
            4. Stakeholder insights""",
            "agent": session_state.crew.empathize_agent,
            "human_input": True,
            "show_file_upload": True
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
            "human_input": True,
            "show_file_upload": True
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
            "human_input": True,
            "show_file_upload": True
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
            "human_input": True,
            "show_file_upload": True
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
            "human_input": True,
            "show_file_upload": True
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
            "agent": session_state.crew.reporting_agent,
            "show_file_upload": False
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
            "agent": session_state.crew.reporting_agent,
            "show_file_upload": False
        },
        "manager_briefing_task" : {
            "name": "Manager Briefing",
            "description" : f"""IMPORTANT: Review and understand the design challenge before coordinating other agents.
    
            DESIGN CHALLENGE: {st.session_state.project_input['challenge']}
            
            CONTEXT: {st.session_state.project_input['context']}
            
            CONSTRAINTS: {str(st.session_state.project_input['constraints'])}
            
            Your job is to ensure all agents focus their work specifically on this challenge, context, and constraints.
            When coordinating their work, continually remind them to refer back to these project parameters.
            """,
            "expected_output" : "Confirmation of understanding the design challenge and plan for coordination",
            "agent" : st.session_state.crew.manager_agent
        }
    
    }

# Initialize session state
def init_session_state():
    if 'project_input' not in st.session_state:
        st.session_state.project_input = None
    
    if 'completed_tasks' not in st.session_state:
        st.session_state.completed_tasks = {}
    
    if 'task_outputs' not in st.session_state:
        st.session_state.task_outputs = {}
    
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
        st.session_state.chat_history = {}
        
    if 'uploaded_pdfs' not in st.session_state:
        st.session_state.uploaded_pdfs = {}
        
    if 'process_logs' not in st.session_state:
        st.session_state.process_logs = []
    
    if 'active_tab_index' not in st.session_state:
        st.session_state.active_tab_index = 0

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
                        
                        # Run the briefing task first
                        briefing_crew = Crew(
                            agents=[st.session_state.crew.manager_agent],
                            tasks=[manager_briefing_task],
                            verbose=True
                        )

                        briefing_result = briefing_crew.kickoff()

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

def display_agent_chat(stage_name, agent_role):
    """Display chat interface for agent stage"""
    st.subheader(f"Chat with {agent_role}")
    
    # Initialize chat history for this stage if not exists
    if stage_name not in st.session_state.chat_history:
        st.session_state.chat_history[stage_name] = []
    
    # Display chat history
    for message in st.session_state.chat_history[stage_name]:
        with st.chat_message(message["role"], avatar=message.get("avatar", "👤" if message["role"] == "user" else "🧠")):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input(f"Ask {agent_role} a question or type 'regenerate' with instructions...", key=f"chat_input_{stage_name}")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history[stage_name].append({
            "role": "user",
            "content": user_input
        })
        
        # Check if the user is asking for regeneration
        if "redo" in user_input.lower() or "regenerate" in user_input.lower():
            # Extract the instructions for regeneration
            instructions = user_input.lower()
            if "regenerate" in instructions:
                instructions = instructions.split("regenerate", 1)[1].strip()
            elif "redo" in instructions:
                instructions = instructions.split("redo", 1)[1].strip()
            
            # Add agent response to chat history
            st.session_state.chat_history[stage_name].append({
                "role": "assistant",
                "content": f"I'll regenerate the content with your instructions: '{instructions}'. Please wait...",
                "avatar": "🧠"
            })
            
            # Add the instructions to stage suggestions
            if stage_name in st.session_state.stage_suggestions:
                st.session_state.stage_suggestions[stage_name] += f"\n- {instructions}"
            else:
                st.session_state.stage_suggestions[stage_name] = f"- {instructions}"
            
            # Record the decision
            if st.session_state.crew:
                st.session_state.crew.decision_tracker.record_decision(
                    stage=f"{stage_name.capitalize()} Regeneration",
                    decision=f"Regenerating content with new instructions",
                    rationale=f"User requested: {instructions}"
                )
            
            # Remove the completed task to force regeneration
            if stage_name in st.session_state.completed_tasks:
                st.session_state.completed_tasks.pop(stage_name, None)
            
            # Rerun to trigger the regeneration
            st.rerun()
        else:
            # Regular question - reference existing content
            task_output = st.session_state.task_outputs.get(stage_name, "No output generated yet.")
            
            # Create a simple prompt using the LLM directly rather than running a full agent task
            if st.session_state.crew and hasattr(st.session_state.crew, "llm"):
                prompt = f"""
                As a {agent_role}, I need to answer a question about my work on this design thinking project.
                
                USER QUESTION: {user_input}
                
                MY PREVIOUS WORK OUTPUT:
                {task_output}
                
                Based ONLY on the information in my previous work output, how should I respond to this question?
                Be conversational, helpful, and refer directly to the content I've already created.
                Do NOT introduce new research or information not present in my work output.
                """
                
                try:
                    # Use direct LLM call instead of creating a new agent task
                    response = st.session_state.crew.llm.call(prompt)
                    agent_response = response
                except Exception as e:
                    agent_response = f"I'm sorry, I had trouble retrieving that information. Error: {str(e)}"
            else:
                agent_response = "I'm sorry, I can't access my previous work right now."
            
            # Add agent response to chat history
            st.session_state.chat_history[stage_name].append({
                "role": "assistant",
                "content": agent_response,
                "avatar": "🧠"
            })
        
        # Rerun to update the UI immediately
        st.rerun()

def generate_agent_response(user_input, stage_name, agent_role):
    """Generate a response from the agent based on user input"""
    # Use the agent from the current stage
    if not st.session_state.crew:
        return "The crew hasn't been initialized yet. Please set up API keys."
    
    # Get task output and context for reference
    task_output = st.session_state.task_outputs.get(stage_name, "No output generated yet.")
    
    # Create a simple task to respond to the user's question
    chat_agent = getattr(st.session_state.crew, f"{stage_name}_agent", None)
    if not chat_agent:
        return f"I'm sorry, but I couldn't find the agent for the {stage_name} stage."
    
    chat_task = Task(
        description=f"""The user has the following question about your work on this design thinking project:
        
        USER QUESTION: {user_input}
        
        PREVIOUS WORK OUTPUT:
        {task_output}
        
        Respond conversationally and directly to the user's question. Be helpful, precise, and friendly.
        Use your expertise as a {agent_role} to provide a valuable response.
        """,
        expected_output="A conversational response to the user's question",
        agent=chat_agent
    )
    
    # Create a temporary crew for just this chat interaction
    chat_crew = Crew(
        agents=[chat_agent],
        tasks=[chat_task],
        verbose=True
    )
    
    try:
        result = chat_crew.kickoff()
        return result.raw
    except Exception as e:
        return f"I'm sorry, I encountered an error while generating a response: {str(e)}"

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

    # Update active tab index to match current stage
    st.session_state.active_tab_index = current_index
    
    # Handle content for all tabs so it persists
    for tab_index, stage in enumerate(task_order):
        with tabs[tab_index]:
            current_task_def = task_definitions[stage]
            
            # Display the agent role
            st.write(f"**Agent:** {current_task_def['agent'].role}")
            
            # Apply any existing suggestions to the task description
            task_description = current_task_def["description"]
            
            if stage in st.session_state.stage_suggestions:
                task_description += f"\n\nHUMAN SUGGESTIONS TO INCORPORATE:\n{st.session_state.stage_suggestions[stage]}"
            
            # Show file upload for applicable stages
            if current_task_def.get("show_file_upload", False):
                st.subheader("Upload Reference PDFs")
                uploaded_files = st.file_uploader(
                    "Upload PDF files for context", 
                    type=["pdf"], 
                    accept_multiple_files=True,
                    key=f"pdf_upload_{stage}"
                )
                
                if uploaded_files:
                    pdf_contents = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Extract text from PDF
                        pdf_text = extract_text_from_pdf(tmp_file_path)
                        pdf_contents.append(pdf_text)
                        
                        # Delete temporary file
                        os.unlink(tmp_file_path)
                    
                    # Store PDF contents for this stage
                    st.session_state.uploaded_pdfs[stage] = pdf_contents
                    
                    st.success(f"{len(uploaded_files)} PDF files processed successfully!")
            
            # Special handling for prototype stage - show solution selection checkboxes
            if stage == "prototype" and "ideate" in st.session_state.completed_tasks and "ideate" in st.session_state.task_outputs:
                st.subheader("Select Solutions to Prototype")
                
                # Parse ideation output to extract solutions
                ideation_output = st.session_state.task_outputs["ideate"]
                
                # Initialize selected_solutions in session state if not present
                if "selected_solutions" not in st.session_state:
                    st.session_state.selected_solutions = {}
                
                # Parse solutions from ideation output
                import re
                solutions = []
                
                # Look for numbered lists, bullet points, or solution headers
                patterns = [
                    r'(?:\d+\.\s*)(.*?)(?=\d+\.|$)',  # Numbered lists
                    r'(?:•\s*)(.*?)(?=•|$)',         # Bullet points
                    r'(?:Solution\s*\d+:?\s*)(.*?)(?=Solution|$)',  # Solution headers
                    r'(?:Idea\s*\d+:?\s*)(.*?)(?=Idea|$)'  # Idea headers
                ]
                
                for pattern in patterns:
                    found_solutions = re.findall(pattern, ideation_output, re.DOTALL)
                    if found_solutions:
                        # Clean up solutions
                        cleaned_solutions = [solution.strip() for solution in found_solutions if solution.strip()]
                        if cleaned_solutions:
                            solutions.extend(cleaned_solutions)
                            break
                
                # If no solutions found with patterns, try to split by lines and look for keywords
                if not solutions:
                    lines = ideation_output.split('\n')
                    for i, line in enumerate(lines):
                        if any(keyword in line.lower() for keyword in ['solution', 'idea', 'concept', 'approach']):
                            if i < len(lines) - 1:
                                solutions.append(line.strip() + " " + lines[i+1].strip())
                            else:
                                solutions.append(line.strip())
                
                # If still no solutions found, use general parsing
                if not solutions:
                    # Split by double line breaks to find paragraphs
                    paragraphs = ideation_output.split('\n\n')
                    for para in paragraphs:
                        if len(para.strip()) > 20 and len(para.strip()) < 500:  # Reasonable solution description length
                            solutions.append(para.strip())
                    
                    # Limit to 5 solutions if we have too many
                    if len(solutions) > 5:
                        solutions = solutions[:5]
                
                # If we still don't have solutions, create placeholder
                if not solutions:
                    st.warning("Couldn't automatically extract solutions from ideation output. Please enter them manually.")
                    solutions = ["Solution 1", "Solution 2", "Solution 3"]
                
                # Store all solutions for use in test stage
                if "all_solutions" not in st.session_state:
                    st.session_state.all_solutions = {}
                st.session_state.all_solutions["prototype"] = solutions
                
                # Show checkboxes for each solution
                selected_solutions = []
                for i, solution in enumerate(solutions):
                    # Truncate if solution is too long
                    display_solution = solution[:200] + "..." if len(solution) > 200 else solution
                    if st.checkbox(display_solution, key=f"prototype_solution_{i}"):
                        selected_solutions.append(solution)
                
                # Store selected solutions
                st.session_state.selected_solutions["prototype"] = selected_solutions
                
                # Add selected solutions to task description when running prototype task
                if selected_solutions:
                    solution_text = "\n\nSELECTED SOLUTIONS TO PROTOTYPE:\n"
                    for i, sol in enumerate(selected_solutions):
                        solution_text += f"{i+1}. {sol}\n"
                    
                    task_description += solution_text
                    
                    # Make sure this is prominently displayed
                    st.info(f"You've selected {len(selected_solutions)} solutions for prototyping")
            
            # Special handling for test stage - show prototyped solution selection checkboxes
            if stage == "test" and "prototype" in st.session_state.completed_tasks and "prototype" in st.session_state.task_outputs:
                st.subheader("Select Prototyped Solutions to Test")
                
                # Get solutions from prototype stage if available, otherwise parse from prototype output
                if "selected_solutions" in st.session_state and "prototype" in st.session_state.selected_solutions:
                    # Use the solutions that were selected for prototyping
                    prototype_solutions = st.session_state.selected_solutions["prototype"]
                else:
                    # Parse prototype output to extract prototyped solutions
                    prototype_output = st.session_state.task_outputs["prototype"]
                    
                    # Try to extract prototyped solutions
                    import re
                    prototype_solutions = []
                    
                    # Look for prototype names, features, specifications
                    patterns = [
                        r'(?:Prototype\s*\d+:?\s*)(.*?)(?=Prototype|$)',  # Prototype headers
                        r'(?:\d+\.\s*)(.*?)(?=\d+\.|$)',  # Numbered lists
                        r'(?:Feature\s*\d+:?\s*)(.*?)(?=Feature|$)'  # Feature headers
                    ]
                    
                    for pattern in patterns:
                        found_solutions = re.findall(pattern, prototype_output, re.DOTALL)
                        if found_solutions:
                            # Clean up solutions
                            cleaned_solutions = [solution.strip() for solution in found_solutions if solution.strip()]
                            if cleaned_solutions:
                                prototype_solutions.extend(cleaned_solutions)
                                break
                
                # Initialize selected test solutions
                if "selected_test_solutions" not in st.session_state:
                    st.session_state.selected_test_solutions = {}
                
                # Show checkboxes for each prototyped solution
                selected_test_solutions = []
                if prototype_solutions:
                    for i, solution in enumerate(prototype_solutions):
                        # Truncate if solution is too long
                        display_solution = solution[:200] + "..." if len(solution) > 200 else solution
                        if st.checkbox(display_solution, key=f"test_solution_{i}", value=True):  # Default selected
                            selected_test_solutions.append(solution)
                else:
                    st.warning("No prototyped solutions found. Please complete the prototype stage first.")
                
                # Store selected solutions for testing
                st.session_state.selected_test_solutions[stage] = selected_test_solutions
                
                # Add selected solutions to task description when running test task
                if selected_test_solutions:
                    test_solution_text = "\n\nSELECTED PROTOTYPES TO TEST:\n"
                    for i, sol in enumerate(selected_test_solutions):
                        test_solution_text += f"{i+1}. {sol}\n"
                    
                    task_description += test_solution_text
                    
                    # Make sure this is prominently displayed
                    st.info(f"You've selected {len(selected_test_solutions)} prototypes for testing")
            
            # Display the task output (if completed)
            if stage in st.session_state.completed_tasks:
                st.success("Task completed!")
                with st.expander("Task Output", expanded=True):
                    st.markdown(st.session_state.task_outputs.get(stage, "No output available"))
                    
                # Display any previous feedback
                if stage in st.session_state.human_feedback:
                    with st.expander("Previous Feedback"):
                        for i, feedback in enumerate(st.session_state.human_feedback[stage]):
                            st.markdown(f"**Feedback {i+1}:** {feedback}")
                
                # Option to revise the task - keep this button visible
                if st.button("Revise this task", key=f"revise_{stage}"):
                    st.session_state.completed_tasks.pop(stage, None)
                    st.rerun()
                
                # Show agent chat interface for all stages if completed
                display_agent_chat(stage, current_task_def['agent'].role)
                
            else:
                # Display the task description and expected output
                st.markdown("### Task Description")
                st.markdown(task_description)
                
                st.markdown("### Expected Output")
                st.markdown(current_task_def["expected_output"])
                
                # Button to run the task
                # For prototype stage, only enable run button when solutions are selected
                run_button_disabled = False
                if stage == "prototype" and "ideate" in st.session_state.completed_tasks:
                    if not st.session_state.selected_solutions.get("prototype", []):
                        st.warning("Please select at least one solution to prototype before running the task.")
                        run_button_disabled = True
                
                # For test stage, only enable run button when prototypes are selected
                if stage == "test" and "prototype" in st.session_state.completed_tasks:
                    if not st.session_state.selected_test_solutions.get(stage, []):
                        st.warning("Please select at least one prototype to test before running the task.")
                        run_button_disabled = True
                
                if not run_button_disabled and st.button("Run Task", key=f"run_{stage}"):
                    with st.spinner(f"Running {current_task_def['name']} task..."):
                        try:
                            # Set context based on completed tasks
                            context_tasks = []
                            
                            # Define context based on design thinking flow
                            if stage == "define" and "empathize" in st.session_state.completed_tasks:
                                context_tasks.append({
                                    "stage": "empathize",
                                    "output": st.session_state.task_outputs["empathize"]
                                })
                                
                            elif stage == "ideate":
                                if "define" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "define",
                                        "output": st.session_state.task_outputs["define"]
                                    })
                                if "empathize" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "empathize",
                                        "output": st.session_state.task_outputs["empathize"]
                                    })
                            
                            # Add other stage conditions as already defined in your code...
                            
                            # Add the context from previous tasks to the task description
                            if context_tasks and len(context_tasks) > 0:
                                task_description += "\n\nPREVIOUS STAGES OUTPUTS:\n"
                                for context in context_tasks:
                                    task_description += f"\n## {context['stage'].upper()} STAGE OUTPUT:\n{context['output']}\n"
                            
                            # Ensure the design challenge, context, and constraints are prominently included
                            task_description = f"""IMPORTANT: Focus on this specific design challenge, context, and constraints.

                DESIGN CHALLENGE: {st.session_state.project_input['challenge']}

                CONTEXT: {st.session_state.project_input['context']}

                CONSTRAINTS: {str(st.session_state.project_input['constraints'])}

                TASK INSTRUCTIONS:
                {task_description}
                """
                            
                            # Now create the task with the updated description
                            task = Task(
                                description=task_description,
                                expected_output=current_task_def["expected_output"],
                                agent=current_task_def["agent"]
                            )
                            
                            
                            # Define context based on design thinking flow
                            if stage == "define" and "empathize" in st.session_state.completed_tasks:
                                context_tasks.append({
                                    "stage": "empathize",
                                    "output": st.session_state.task_outputs["empathize"]
                                })
                                
                            elif stage == "ideate":
                                if "define" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "define",
                                        "output": st.session_state.task_outputs["define"]
                                    })
                                if "empathize" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "empathize",
                                        "output": st.session_state.task_outputs["empathize"]
                                    })
                                
                            elif stage == "prototype":
                                if "ideate" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "ideate",
                                        "output": st.session_state.task_outputs["ideate"]
                                    })
                                if "define" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "define",
                                        "output": st.session_state.task_outputs["define"]
                                    })
                                if "selected_solutions" in st.session_state and "prototype" in st.session_state.selected_solutions:
                                                                    
                                    selected_sols = st.session_state.selected_solutions["prototype"]
                                    if selected_sols:
                                        task_description += f"\n\nIMPORTANT: You MUST create detailed prototype descriptions for ALL {len(selected_sols)} selected solutions listed above. Ensure each solution is given equal attention and detail."

                            elif stage == "test":
                                if "prototype" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "prototype",
                                        "output": st.session_state.task_outputs["prototype"]
                                    })
                                if "ideate" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "ideate",
                                        "output": st.session_state.task_outputs["ideate"]
                                    })
                                if "selected_test_solutions" in st.session_state and stage in st.session_state.selected_test_solutions:
                                    selected_tests = st.session_state.selected_test_solutions[stage]
                                    if selected_tests:
                                        task_description += f"\n\nIMPORTANT: You MUST create comprehensive testing protocols for ALL {len(selected_tests)} selected prototypes listed above. Each prototype must have its own testing approach with equal detail and consideration."
                                
                            elif stage == "decisions":
                                # Add all previous tasks as context
                                for name in ["empathize", "define", "ideate", "prototype", "test"]:
                                    if name in st.session_state.completed_tasks:
                                        context_tasks.append({
                                            "stage": name,
                                            "output": st.session_state.task_outputs[name]
                                        })
                                
                            elif stage == "report":
                                for name in st.session_state.completed_tasks:
                                    if name in st.session_state.task_outputs:
                                        context_tasks.append({
                                            "stage": name,
                                            "output": st.session_state.task_outputs[name]
                                        })
                            
                            # Get PDF contents for this stage
                            pdf_contents = st.session_state.uploaded_pdfs.get(stage, [])
                            
                            # Create a list of available agents for the hierarchical process
                            available_agents = []  # Start with manager
                            
                            # Add agents based on completed stages and current stage
                            if stage == "empathize" or "empathize" in st.session_state.completed_tasks:
                                available_agents.append(st.session_state.crew.empathize_agent)
                            if stage == "define" or "define" in st.session_state.completed_tasks:
                                available_agents.append(st.session_state.crew.define_agent)
                            if stage == "ideate" or "ideate" in st.session_state.completed_tasks:
                                available_agents.append(st.session_state.crew.ideate_agent)
                            if stage == "prototype" or "prototype" in st.session_state.completed_tasks:
                                available_agents.append(st.session_state.crew.prototype_agent)
                            if stage == "test" or "test" in st.session_state.completed_tasks:
                                available_agents.append(st.session_state.crew.test_agent)
                            if stage == "decisions" or "decisions" in st.session_state.completed_tasks:
                                available_agents.append(st.session_state.crew.reporting_agent)
                            if stage == "report" or "report" in st.session_state.completed_tasks:
                                available_agents.append(st.session_state.crew.reporting_agent)
                                
                            # Create hierarchical crew for this task
                            hierarchical_crew = Crew(
                                agents=available_agents,
                                tasks=[task],
                                process=Process.hierarchical,
                                manager_agent=st.session_state.crew.manager_agent,
                                verbose=True
                            )
                            
                            # Run the task with hierarchical process
                            with st.status("Running task with manager supervision..."):
                                st.write(f"Manager agent is coordinating task execution for {current_task_def['name']}...")
                                result = hierarchical_crew.kickoff(
                                        inputs={
                                        "project_input": st.session_state.project_input,
                                        "challenge": st.session_state.project_input['challenge'],
                                        "context": st.session_state.project_input['context'],
                                        "constraints": st.session_state.project_input['constraints'],
                                        "context_tasks": context_tasks if context_tasks else [],
                                        "pdf_contents": pdf_contents if pdf_contents else []
                                    }
                                )
                                st.write("Task completed successfully!")
                            
                            # Extract any manager logs/interactions for process logs
                            manager_logs = []
                            try:
                                # Extract manager logs from the result if available
                                if hasattr(result, 'log') and result.log:
                                    manager_logs = result.log
                                # Add a basic log entry if none found
                                if not manager_logs:
                                    manager_logs.append(f"Manager coordinated the {current_task_def['name']} task execution")
                            except:
                                manager_logs = [f"Manager supervised {current_task_def['name']} task"]
                            
                            # Add to process logs
                            st.session_state.process_logs.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "stage": stage,
                                "action": f"Executed {current_task_def['name']} task",
                                "manager_notes": manager_logs
                            })
                            
                            # Store the completed task
                            st.session_state.completed_tasks[stage] = task
                            
                            # Store the task output separately for easy access
                            st.session_state.task_outputs[stage] = result.raw
                            
                            # Add task to history
                            st.session_state.task_history.append({
                                "stage": stage,
                                "agent": current_task_def['agent'].role,
                                "output": result.raw
                            })
                                
                            st.rerun()

                        except Exception as e:
                                
                                st.error(f"Error executing task: {e}")
    
    # Navigation controls
    st.subheader("Navigation")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Previous button
        if current_index > 0:
            if st.button("⬅️ Previous Stage", key="prev_stage_button"):
                prev_stage_index = current_index - 1
                st.session_state.current_stage = task_order[prev_stage_index]
                # Update active tab index
                st.session_state.active_tab_index = prev_stage_index
                st.rerun()
    
    with col3:
        # Next button
        if current_index < len(task_order) - 1:
            if st.button("Next Stage ➡️", key="next_stage_button"):
                # Set the next stage
                next_stage_index = current_index + 1
                st.session_state.current_stage = task_order[next_stage_index]
                # Update the active tab index
                st.session_state.active_tab_index = next_stage_index
                st.rerun()

    with col2:
        # Jump to any stage
        jump_options = ["Jump to..."] + [task_definitions[stage]["name"] for stage in task_order]
        jump_to = st.selectbox("", jump_options, index=0)
        
        if jump_to != "Jump to...":
            jump_index = [task_definitions[stage]["name"] for stage in task_order].index(jump_to)
            target_stage = task_order[jump_index]
            if st.button(f"Go to {jump_to}", key="jump_button"):
                st.session_state.current_stage = target_stage
                # Update active tab index
                st.session_state.active_tab_index = jump_index
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
        main_tabs = st.tabs(["Challenge Setup", "Design Thinking Process", "Decision Log"])
        
        with main_tabs[0]:
            setup_challenge()
            display_challenge()
            
        with main_tabs[1]:
            if st.session_state.project_input:
                run_design_thinking_process()
            else:
                st.info("Please set up a design challenge first in the Challenge Setup tab.")
                
        with main_tabs[2]:
            display_decision_log()

if __name__ == "__main__":
    main()    
