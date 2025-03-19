__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import base64
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
from streamlit.components.v1 import html
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, RagTool, PDFSearchTool
import tempfile
from fpdf import FPDF
import markdown
import tempfile
from io import BytesIO 

def create_pdf_from_markdown(markdown_text, title):
    """Convert markdown to PDF and return the PDF as bytes"""
    class CustomPDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            # Encode title to avoid Unicode issues
            safe_title = title.encode('ascii', 'replace').decode('ascii')
            self.cell(0, 10, safe_title, 0, 1, 'C')
            self.ln(10)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    try:
        # Initialize PDF
        pdf = CustomPDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 11)
        
        # Convert markdown to plain text and handle encoding
        html = markdown.markdown(markdown_text)
        
        # Clean up HTML and handle special characters
        clean_text = html.replace('<p>', '').replace('</p>', '\n')
        clean_text = clean_text.replace('<strong>', '').replace('</strong>', '')
        clean_text = clean_text.replace('"', '"').replace('"', '"')
        clean_text = clean_text.replace(''', "'").replace(''', "'")
        clean_text = clean_text.replace('–', '-').replace('—', '-')
        clean_text = clean_text.replace('•', '*')
        
        # Convert to ASCII, replacing unsupported characters
        safe_text = clean_text.encode('ascii', 'replace').decode('ascii')
        
        # Write content with proper line breaks
        pdf.set_auto_page_break(auto=True, margin=15)
        for line in safe_text.split('\n'):
            if line.strip():
                try:
                    pdf.multi_cell(0, 10, line.strip())
                except:
                    # If a line fails, try writing it character by character
                    for char in line:
                        try:
                            pdf.write(10, char)
                        except:
                            pdf.write(10, '?')
                    pdf.ln()
        
        # Save to bytes
        with BytesIO() as bytes_file:
            pdf.output(bytes_file)
            return bytes_file.getvalue()
            
    except Exception as e:
        # Create simple PDF with error message
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 12)
        error_msg = f"Error generating PDF. Please try again.\nError: {str(e)}"
        pdf.multi_cell(0, 10, error_msg)
        
        with BytesIO() as bytes_file:
            pdf.output(bytes_file)
            return bytes_file.getvalue()

def sanitize_task_description(description):
    """Clean up task description to avoid template variable errors"""
    # Remove any potential template variable markers
    description = description.replace('{{', '{').replace('}}', '}')
    description = description.replace('{%', '').replace('%}', '')
    # Replace quotes with straight quotes
    description = description.replace('"', '"').replace('"', '"')
    description = description.replace(''', "'").replace(''', "'")
    return description

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

class EnhancedCitationTrackingSerperTool(SerperDevTool):
    def _run(self, query: str) -> str:
        # Get context before running query
        context = {
            'stage': getattr(st.session_state, 'current_stage', 'unknown'),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'previous_task_output': st.session_state.task_outputs.get(
                st.session_state.current_stage, None
            ),
            'related_queries': [
                q['query'] for q in st.session_state.search_queries.get(
                    st.session_state.current_stage, []
                )
            ]
        }
        
        result = super()._run(query)
        
        # Store enhanced query info
        query_info = {
            'query': query,
            'context': context,
            'result_summary': result[:500] + "..." if len(result) > 500 else result
        }
        
        if context['stage'] in st.session_state.search_queries:
            st.session_state.search_queries[context['stage']].append(query_info)
            
        # Process citations as before with enhanced context
        try:
            self._process_citations(query, result, context)
        except Exception as e:
            print(f"Error processing citations: {e}")
            
        return result

    def _process_citations(self, query, result, context):
        import json
        data = json.loads(self.search.results(query))
        citations = []
        
        if 'organic' in data:
            for item in data['organic']:
                citation = {
                    'query': query,
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'stage': context['stage'],
                    'timestamp': context['timestamp'],
                    'context': context
                }
                citations.append(citation)
        
        if 'citations' not in st.session_state:
            st.session_state.citations = {}
        
        st.session_state.citations[query] = citations

class CitationTrackingSerperTool(SerperDevTool):
    def _run(self, query: str) -> str:
        result = super()._run(query)
        
        # Track the current stage and query
        current_stage = getattr(st.session_state, 'current_stage', 'unknown')
        
        # Store query in session state
        if current_stage in st.session_state.search_queries:
            st.session_state.search_queries[current_stage].append({
                'query': query,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Extract citations from the result
        try:
            import json
            data = json.loads(self.search.results(query))
            citations = []
            
            # Extract organic search results
            if 'organic' in data:
                for item in data['organic']:
                    citation = {
                        'query': query,
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'stage': current_stage,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    citations.append(citation)
            
            # Store citations in session state
            if 'citations' not in st.session_state:
                st.session_state.citations = {}
            
            if query not in st.session_state.citations:
                st.session_state.citations[query] = citations
                
        except Exception as e:
            print(f"Error tracking citations: {e}")
            
        return result

class DesignThinkingCrew:
    """Design Thinking crew for user-centered problem solving with feedback loops"""
    
    def __init__(self, api_keys: Dict[str, str], model_name: str = ""):
        """Initialize the Design Thinking Crew with necessary API keys and model selection"""
        # Set Serper API key in environment
        os.environ["SERPER_API_KEY"] = api_keys.get("serper")
        
        # Initialize LLM based on selected model
        if "claude" in model_name.lower():
            self.llm = LLM(
                model=model_name,  # anthropic/claude-3-7-sonnet-20250219
                api_key=api_keys.get("claude")
            )
        elif "groq" in model_name.lower():
            self.llm = LLM(
                model=model_name,  # groq/deepseek-r1-distill-llama-70b
                api_key=api_keys.get("groq")
            )

        elif "deepseek" in model_name.lower():
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

        # Initialize the RAG tool for design thinking knowledge
        self.design_thinking_knowledge = RagTool(
            config=dict(
                llm=dict(
                    provider="google",  # Using Google (Gemini) as default
                    config=dict(
                        model="gemini/gemini-2.0-flash",
                        api_key=api_keys.get("gemini")
                    ),
                ),
                embedder=dict(
                    provider="google",
                    config=dict(
                        model="gemini-embedding-exp-03-07",
                        task_type="retrieval_document",
                    ),
                ),
            ),
            files=["The Design Thinking Playbook.pdf"],  # This will be set dynamically
            name="Design Thinking Guide",
            description="A comprehensive guide for design thinking methodology and best practices"
        )


               
        # Initialize agents with refined definitions
        self.empathize_agent = Agent(
            role="Empathy Researcher",
            goal="Uncover the deepest human needs, pain points, and motivations through immersive research to build genuine empathy and establish a foundation for user-centered solutions",
            backstory="""You are a world-class ethnographic researcher who has led user research for groundbreaking products at IDEO, IBM Design, 
            and top innovation consultancies. With formal training in anthropology and behavioral psychology, you've developed techniques that 
            reveal what users truly need—not just what they say they want. You've masterfully conducted hundreds of contextual inquiries, 
            in-depth interviews, and immersive observations across diverse cultures and domains. You're known for your remarkable ability to 
            spot behavioral patterns that others miss and to extract insights from seemingly ordinary interactions. Your empathy maps and user 
            journey visualizations have become standard tools in the field. You approach each conversation with genuine curiosity, asking probing 
            questions that reveal underlying motivations rather than surface-level preferences. You believe deeply that empathy is 
            the foundation of all meaningful innovation.""",
            tools=[self.search_tool],
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        self.define_agent = Agent(
            role="Problem Definition Specialist",
            goal="Transform complex research data into laser-focused problem statements that reveal core user needs, challenge assumptions, and create a foundation for breakthrough innovation",
            backstory="""You are a renowned synthesis expert who has led problem framing workshops at Stanford's d.school and helped Fortune 100 companies 
            redefine seemingly intractable challenges into breakthrough opportunities. Your command of frameworks like Jobs-to-be-Done and Outcome-Driven 
            Innovation has helped teams see beyond symptoms to root causes. You've developed a reputation for ruthlessly cutting through informational 
            noise to identify the essential problem worth solving. Your ability to craft "How Might We" statements has launched numerous successful 
            products by framing challenges at exactly the right level—neither too broad nor too narrow. You've created detailed user personas and journey 
            maps that have become legendary in the field for their depth and accuracy. Teams regularly cite your problem definitions as the pivotal 
            moment that redirected their efforts toward truly meaningful solutions. You believe that a well-defined problem is already half-solved.""",
            tools=[self.search_tool, self.design_thinking_knowledge],  # Added internet search tool
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        self.ideate_agent = Agent(
            role="Innovation Facilitator",
            goal="Orchestrate breakthrough ideation processes that generate radical, diverse solutions through creative tension, cognitive diversity, and systematic exploration of possibility spaces",
            backstory="""You are an internationally recognized innovation expert who has facilitated ideation sessions at LEGO's Creative Play Lab,
            Google X, and top innovation hubs worldwide. Your background spans design, engineering, and creative psychology, giving you a unique 
            approach to idea generation. You've mastered dozens of ideation techniques beyond common brainstorming—from systematic inventive thinking
            and biomimicry to SCAMPER, design analogies, and provocation techniques. You've helped teams generate over 10,000 concepts across industries, 
            many evolving into market-leading products. You excel at creating the psychological safety needed for wild ideas while maintaining the 
            productive tension that drives innovative thinking. You know precisely when to diverge broadly and when to begin converging toward actionable 
            concepts. You've developed proprietary methods for idea evaluation that balance feasibility, viability, and desirability. 
            You believe creativity is both art and science—requiring both spontaneous connections and rigorous exploration of solution spaces.""",
            tools=[self.search_tool, self.design_thinking_knowledge],  # Added internet search tool
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        self.prototype_agent = Agent(
            role="Prototyping Specialist",
            goal="Transform abstract concepts into tangible, testable experiences using the minimal resources needed to validate critical assumptions and communicate the essence of solutions",
            backstory="""You are a prototype virtuoso who has led rapid prototyping labs at Frog Design, Apple, and MIT Media Lab. 
            Your diverse background spans industrial design, software development, and experience design, allowing you to prototype across physical, 
            digital, and service domains. You've pioneered the philosophy of "minimum viable fidelity"—creating prototypes with precisely the right level 
            of polish needed to test key assumptions without wasting resources. Your prototyping matrix framework helps teams determine whether to build 
            paper prototypes, digital wireframes, functional mockups, or experience simulations based on what they need to learn. You've developed a 
            reputation for ingenious resource utilization—turning ordinary materials into compelling prototypes in hours rather than weeks. 
            You've trained teams to iterate rapidly through multiple prototype generations, each testing specific aspects of the solution. 
            You believe prototypes are not just validation tools but boundary objects that create shared understanding across stakeholders with 
            different perspectives.""",
            tools=[self.search_tool, self.design_thinking_knowledge],  # Added internet search tool
            verbose=True,
            llm=self.llm,
            allow_delegation=False  # Prototyping agent cannot delegate tasks
        )
        
        self.test_agent = Agent(
            role="User Testing Coordinator",
            goal="Design rigorous yet flexible testing protocols that extract maximum learning from user interactions, validate critical assumptions, and guide iterative refinement toward solutions users truly value",
            backstory="""You are a renowned testing methodologist who has led user research labs at Nielsen Norman Group, established testing protocols 
            for breakthrough products at Tesla, and taught evaluation methods at Carnegie Mellon's HCI program. Your dual background in experimental 
            psychology and interaction design gives you unique insight into how to design tests that yield meaningful data. You've mastered diverse 
            testing approaches—from guerrilla testing and remote unmoderated studies to controlled experiments and longitudinal field studies. 
            You're known for your ability to craft testing protocols that extract maximum insight with minimal interference in natural user behavior. 
            You've developed sophisticated frameworks for analyzing qualitative feedback that reveal patterns invisible to most observers. Your ability 
            to distinguish between what users say, what they do, and what they truly need has saved countless teams from building solutions based on 
            misleading feedback. You believe that testing is not a validation exercise but a learning journey, and you excel at helping teams embrace 
            and act on uncomfortable findings rather than explaining them away.""",
            tools=[self.search_tool, self.design_thinking_knowledge],  # Added internet search tool
            verbose=True,
            llm=self.llm,
            allow_delegation=False
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
            goal="Craft compelling, evidence-based narratives that document the design journey, communicate insights with crystal clarity, and translate complex design decisions into stakeholder-appropriate documentation",
            backstory="""You are an acclaimed design documentarian who has created case studies for the Design Management Institute, documented 
            breakthrough innovation processes at Mayo Clinic's Center for Innovation, and authored respected publications on design communication. 
            Your background in both design thinking and data visualization gives you a unique ability to tell stories that balance qualitative insights 
            with quantitative evidence. You've pioneered innovative documentation formats that capture the messiness of real design processes while 
            providing clear throughlines that stakeholders can follow. Your reports have become known for their strategic use of visuals, quotes, and 
            artifacts that bring the design journey to life. You've developed frameworks for communicating design decisions that resonate with different 
            stakeholder perspectives—from C-suite executives and engineering teams to marketing departments and end users. Your documentation has 
            repeatedly secured additional funding for projects by clearly connecting design decisions to business outcomes. You believe that great 
            design work is only as valuable as its communication, and you're passionate about creating living documents that continue to provide value 
            long after the initial design phase.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
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
            tools=[self.search_tool, self.design_thinking_knowledge],  # Add search tool to research the domain if needed
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
def generate_enhanced_research_report():
    report = "# Design Thinking Research Report\n\n"
    
    # Executive Summary
    report += "## Executive Summary\n"
    report += f"Research conducted across {sum(1 for queries in st.session_state.search_queries.values() if queries)} stages "
    report += f"with {sum(len(queries) for queries in st.session_state.search_queries.values())} total queries.\n\n"
    
    # Research Methodology
    report += "## Research Methodology\n\n"
    for stage, queries in st.session_state.search_queries.items():
        if queries:
            report += f"### {stage.capitalize()} Stage\n\n"
            report += f"**Number of Queries:** {len(queries)}\n\n"
            report += "#### Key Research Questions:\n"
            for query in queries:
                report += f"- {query['query']}\n"
                if query['query'] in st.session_state.citations:
                    report += "  Key findings:\n"
                    for citation in st.session_state.citations[query['query']][:3]:
                        report += f"  - {citation['title']}\n"
            report += "\n"
    
    # Source Analysis
    report += "## Source Analysis\n\n"
    domain_analysis = {}
    for citations in st.session_state.citations.values():
        for citation in citations:
            domain = urlparse(citation['link']).netloc
            if domain not in domain_analysis:
                domain_analysis[domain] = {
                    'count': 0,
                    'titles': set(),
                    'stages': set()
                }
            domain_analysis[domain]['count'] += 1
            domain_analysis[domain]['titles'].add(citation['title'])
            domain_analysis[domain]['stages'].add(citation['stage'])
    
    # Add domain analysis to report
    for domain, analysis in sorted(domain_analysis.items(), 
                                 key=lambda x: x[1]['count'], 
                                 reverse=True):
        report += f"### {domain}\n"
        report += f"- Referenced {analysis['count']} times\n"
        report += f"- Used in stages: {', '.join(sorted(analysis['stages']))}\n"
        report += "- Key articles:\n"
        for title in list(analysis['titles'])[:3]:
            report += f"  - {title}\n"
        report += "\n"
    
    return report

def run_task(self, task_name, task, project_input, context_tasks=None, pdf_contents=None):
    """Run a single task and return its result"""
    try:
        if context_tasks:
            task.context = context_tasks

            # Enhance task description with design thinking methodology prompt
            task.description = f"""IMPORTANT: Reference the design thinking guide to inform your approach.
            Consider multiple methodologies and frameworks when addressing this challenge.
            Justify your chosen approaches based on established design thinking principles.
            
            {task.description}
            
            When developing your response:
            1. Review relevant methodologies from the design thinking guide
            2. Consider multiple alternative approaches
            3. Select and justify methods based on the specific context
            4. Structure your output according to proven frameworks
            """

        # Incorporate PDF contents into task description if available
        if pdf_contents and len(pdf_contents) > 0:
            pdf_context = "\n\nREFERENCE PDF DOCUMENTS:\n"
            for i, content in enumerate(pdf_contents):
                # Limit content length to avoid token issues
                truncated_content = content[:10000] + "..." if len(content) > 10000 else content
                pdf_context += f"\nPDF DOCUMENT {i+1}:\n{truncated_content}\n\n"
            
            task.description += pdf_context
        
        # Create temporary crew for this task
        temp_crew = Crew(
            agents=[task.agent],
            tasks=[task],
            verbose=True
        )
        
        # Add extensive error handling and validation
        try:
            result = temp_crew.kickoff(
                inputs={
                    "project_input": project_input,
                    "context_tasks": context_tasks if context_tasks else [],
                    "pdf_contents": pdf_contents,
                    "reference_guide": True  # Flag to indicate design thinking guide should be referenced
                }
            )    
            return result
            
        except Exception as e:
            if "Invalid response from LLM" in str(e) or "empty response" in str(e):
                # Try with simplified prompt
                simplified_task = Task(
                    description=f"""SIMPLIFIED RETRY - Previous attempt failed. Please provide a basic response for:
                    
                    CHALLENGE: {project_input['challenge']}
                    TASK: {task.description[:500]}... (truncated)
                    
                    Focus on core requirements for {task_name} stage.
                    """,
                    expected_output=task.expected_output,
                    agent=task.agent
                )
                
                result = temp_crew.kickoff(inputs=project_input)
                
                if result and hasattr(result, 'raw') and result.raw:
                    return result
                else:
                    raise ValueError(f"Failed to get valid response after retry for {task_name}")
            else:
                raise e
                
    except Exception as e:
        st.error(f"Error executing task: {str(e)}")
        from crewai.tasks.task_output import TaskOutput
        return TaskOutput(
            task_id=task_name,
            raw=f"Task execution failed: {str(e)}. Please try again or contact support if the issue persists.",
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
        return f"[Error extracting PDF content: {str(e)}]"

def prepare_research_summary():
    summary = "\n## Research Methodology\n\n"
    summary += "### Search Queries by Stage\n\n"
    
    for stage, queries in st.session_state.search_queries.items():
        if queries:
            summary += f"\n#### {stage.capitalize()} Stage Queries:\n"
            for q in queries:
                summary += f"- {q['timestamp']}: `{q['query']}`\n"
    
    summary += "\n### Referenced Sources\n\n"
    seen_links = set()
    for query, citations in st.session_state.citations.items():
        summary += f"\nSources for query: `{query}`\n"
        for citation in citations:
            if citation['link'] not in seen_links:
                summary += f"- [{citation['title']}]({citation['link']})\n"
                summary += f"  - Stage: {citation['stage']}\n"
                summary += f"  - Time: {citation['timestamp']}\n"
                summary += f"  - Context: _{citation['snippet']}_\n\n"
                seen_links.add(citation['link'])
    
    return summary

# Define task definitions globally so it can be accessed by multiple functions
def get_task_definitions(session_state):
    """Get task definitions for the design thinking process"""
    if not session_state.crew or not session_state.project_input:
        return {}
    
    return {
        "empathize": {
            "name": "Empathy",
            "description": f"""Uncover systemic patterns and latent needs for: {session_state.project_input['challenge']}

            METHODOLOGY REQUIREMENTS:
            1. Research Methods Selection
                - Review and select appropriate qualitative research methods from the guide
                - Justify method selection based on context and constraints
                - Plan for multiple data collection approaches
            
            2. User Understanding Framework
                - Apply structured empathy mapping techniques
                - Implement journey mapping methodologies
                - Use behavioral analysis frameworks
            
            3. Data Collection Approach
                - Design interview protocols based on best practices
                - Plan observation methodologies
                - Structure feedback collection methods
            
            4. Synthesis Preparation
                - Setup pattern recognition frameworks
                - Prepare insight categorization systems
                - Plan for data triangulation
            
            5. Documentation Strategy
                - Implement structured note-taking methods
                - Plan for multimedia documentation
                - Setup insight capture frameworks

            CONTEXTUAL CONSIDERATIONS:
            - Challenge Context: {session_state.project_input['context']}
            - Key Constraints: {str(session_state.project_input['constraints'])}
            
            EXPECTED DELIVERABLES:
            1. Comprehensive Empathy Map
            2. User Journey Documentation
            3. Key Insights Matrix
            4. Behavioral Patterns Analysis
            5. Emotional Need States Mapping""",
            "expected_output": """Strategic Empathy Dossier containing:

            **Core Empathy Map** (preserving requested structure with enhanced depth):
            1. SAYS: 
            - Direct quotes + linguistic analysis of metaphors/emphasis
            - Contradictions between different interview moments
            2. THINKS: 
            - Unarticulated assumptions
            - Cognitive biases impacting decisions
            3. USER NEEDS: 
            - Hierarchy of needs (Basic > Emotional > Aspirational)
            - "Progress Struggles" (JTBD framework)
            4. FEELS: 
            - Emotional journey mapping
            - Frustration/desire heat mapping
            5. DOES: 
            - Observed behaviors vs stated behaviors
            - Workaround solutions analysis

            **Supplemental Insights:**
            1. Stakeholder Influence Network Map
            2. Contradiction Matrix (SAYS vs DOES)
            3. Emotional Journey Map
            4. Emerging Opportunity Signals
             Format: Visual ecosystem map + narrative insights report""",
            "agent": session_state.crew.empathize_agent,
            "human_input": True,
            "show_file_upload": True
        },  
        "define": {
            "name": "Define",
            "description": f"""Transform research insights into actionable problem definitions for: {session_state.project_input['challenge']}

            METHODOLOGY REQUIREMENTS:
            1. Insight Analysis Framework
                - Apply structured synthesis methods
                - Use problem framing techniques
                - Implement insight clustering approaches
            
            2. Problem Statement Development
                - Utilize "How Might We" frameworks
                - Apply root cause analysis methods
                - Use problem reframing techniques
            
            3. User Needs Mapping
                - Implement needs hierarchy frameworks
                - Use jobs-to-be-done methodology
                - Apply outcome-driven innovation approaches
            
            4. Opportunity Space Definition
                - Use opportunity mapping frameworks
                - Apply constraint analysis methods
                - Implement possibility thinking approaches
            
            5. Success Criteria Development
                - Use metric development frameworks
                - Apply validation criteria methods
                - Implement measurement planning approaches

            CONTEXTUAL CONSIDERATIONS:
            - Challenge Context: {session_state.project_input['context']}
            - Key Constraints: {str(session_state.project_input['constraints'])}
            
            EXPECTED DELIVERABLES:
            1. Problem Statement Matrix
            2. User Needs Framework
            3. Opportunity Spaces Map
            4. Success Criteria Definition
            5. Validation Framework""",

            "expected_output": """1. Multilayer Problem Statements:
            - Surface-level symptoms vs systemic root causes
            - JTBD-driven "Progress Struggles" analysis
            2. 2-Paradox Personas:
            - Core needs vs contextual constraints
            - Decision-making tension maps
            3. Innovation Criteria Matrix:
            - Desirability/Viability/Feasibility weightings
            - Constraint transformation opportunities
            4. How-Might-We Opportunity Landscape:
            - HMW spectrum (Incremental <-> Transformational)
            - Adjacent possibility spaces""",
            "agent": session_state.crew.define_agent,
            "human_input": True,
            "show_file_upload": True
        },
        "ideate": {
            "name": "Ideate",
            "description": f"""Generate diverse solution concepts for: {session_state.project_input['challenge']}

            METHODOLOGY REQUIREMENTS:
            1. Ideation Method Selection
                - Choose appropriate brainstorming techniques
                - Plan divergent thinking approaches
                - Structure convergent methods
            
            2. Creative Framework Application
                - Implement lateral thinking tools
                - Use analogical thinking methods
                - Apply systematic innovation techniques
            
            3. Solution Development Process
                - Use concept development frameworks
                - Apply idea building methods
                - Implement combination techniques
            
            4. Evaluation Framework
                - Use idea assessment criteria
                - Apply feasibility analysis methods
                - Implement impact evaluation approaches
            
            5. Documentation System
                - Use idea capture frameworks
                - Apply organization methods
                - Implement tracking systems

            CONTEXTUAL CONSIDERATIONS:
            - Challenge Context: {session_state.project_input['context']}
            - Key Constraints: {str(session_state.project_input['constraints'])}
            
            EXPECTED DELIVERABLES:
            1. Solution Concepts Portfolio
            2. Innovation Opportunity Map
            3. Concept Evolution Framework
            4. Evaluation Matrix
            5. Development Roadmap""",

            "expected_output": """
            Around 5-8 solution concepts, each with:
            Evaluation of each idea to be a table breaking down the comparisons and trade-offs
            1. Solution Archetypes:
            - Core concept
            - Variant possibilities
            - Adjacent applications
            2. Concept Evolution Pathways:
            - Maturity timeline (Now/Near/Far)
            - Dependency mapping
            3. Innovation Impact Matrix:
            - User value vs implementation complexity
            - First-principles breakthrough potential
            4. Hybridization Opportunities:
            - Concept combination matrix
            - Cross-domain inspiration"""
            ,
            "agent": session_state.crew.ideate_agent,
            "human_input": True,
            "show_file_upload": True
        },
        "prototype": {
            "name": "Prototype",
            "description": f"""Create testable manifestations of selected solutions for: {session_state.project_input['challenge']}

            METHODOLOGY REQUIREMENTS:
            1. Prototype Strategy Development
                - Select appropriate fidelity levels
                - Plan iteration approaches
                - Structure testing methods
            
            2. Creation Framework
                - Use rapid prototyping techniques
                - Apply minimum viable product methods
                - Implement feedback integration approaches
            
            3. Testing Preparation
                - Design validation frameworks
                - Plan user interaction methods
                - Structure feedback collection
            
            4. Iteration Planning
                - Use improvement frameworks
                - Apply refinement methods
                - Implement evolution approaches
            
            5. Documentation Process
                - Use capture methods
                - Apply learning frameworks
                - Implement tracking systems

            CONTEXTUAL CONSIDERATIONS:
            - Challenge Context: {session_state.project_input['context']}
            - Key Constraints: {str(session_state.project_input['constraints'])}
            
            EXPECTED DELIVERABLES:
            1. Prototype Specifications
            2. Testing Framework
            3. Iteration Plan
            4. Documentation System
            5. Evolution Strategy""",

            "expected_output": """1. Prototyping Strategy:
            - Key assumptions hierarchy
            - Validation sequencing
            2. Multi-Fidelity Plan:
            - Paper prototype -> Digital mockup -> Wizard-of-Oz
            3. Resourceful Materials Matrix:
            - Low-cost high-impact material substitutions
            - Digital twin simulation options
            4. Failure Injection Scenarios:
            - Controlled stress tests
            - Edge case exploration""",
            "agent": session_state.crew.prototype_agent,
            "human_input": True,
            "show_file_upload": True
        },
        "test": {
            "name": "Test",
            "description": f"""Validate prototype effectiveness and gather user feedback for: {session_state.project_input['challenge']}

            METHODOLOGY REQUIREMENTS:
            1. Testing Strategy Development
                - Select validation methods
                - Plan user engagement
                - Structure feedback collection
            
            2. User Testing Framework
                - Use interaction protocols
                - Apply observation methods
                - Implement feedback systems
            
            3. Data Collection Process
                - Design capture methods
                - Plan analysis approaches
                - Structure insight gathering
            
            4. Evaluation System
                - Use assessment frameworks
                - Apply success metrics
                - Implement improvement tracking
            
            5. Iteration Planning
                - Use refinement methods
                - Apply evolution frameworks
                - Implement development paths

            CONTEXTUAL CONSIDERATIONS:
            - Challenge Context: {session_state.project_input['context']}
            - Key Constraints: {str(session_state.project_input['constraints'])}
            
            EXPECTED DELIVERABLES:
            1. Testing Protocol
            2. Feedback Framework
            3. Analysis System
            4. Evaluation Matrix
            5. Iteration Strategy""",

            "expected_output": """1. Mixed-Methods Protocol:
            - Behavioral observation grids
            - Emotion tracking scales
            - Cognitive walkthroughs
            2. Contradiction Analysis Framework:
            - Say/Do gaps
            - Conscious/Unconscious mismatches
            3. Insight Extraction Matrix:
            - Surface reactions
            - Latent needs signals
            - Ecosystem impacts
            4. Pivot/Persevere Decision Tree:
            - Confidence thresholds
            - Iteration priority stack""",
            "agent": session_state.crew.test_agent,
            "human_input": True,
            "show_file_upload": True
        },
        "decisions": {
            "name": "Decision Archaeology",
            "description": f"""Document and analyze the decision-making process throughout the design thinking journey for: {st.session_state.project_input['challenge']}

            COMPREHENSIVE DECISION ANALYSIS REQUIREMENTS:

            1. Agent Decision Making Process
                - Document each agent's key decisions
                - Analyze decision rationale and methodology
                - Map the decision trees and alternative paths considered
                - Identify critical decision points and their impact
            
            2. Cross-Stage Decision Flow
                - Track how decisions in each stage influenced subsequent stages
                - Document information flow between stages
                - Analyze the ripple effects of key decisions
                - Map dependencies between stage decisions
            
            3. Manager Coordination Analysis
                - Document how the manager agent coordinated decisions
                - Analyze the effectiveness of inter-agent communication
                - Map the delegation and task distribution process
                - Evaluate the manager's role in decision optimization
            
            4. Decision Rationale Documentation
                - Capture detailed reasoning behind each major decision
                - Document alternative options considered
                - Analyze trade-offs and their implications
                - Track how user needs influenced decisions
            
            5. Context Integration Analysis
                - Document how context was passed between stages
                - Analyze how each stage built upon previous decisions
                - Map the evolution of understanding across stages
                - Track how insights influenced decision-making

            PREVIOUS STAGE OUTPUTS AND DECISIONS:
            {chr(10).join([f"## {stage.upper()} STAGE DECISIONS:" + st.session_state.task_outputs.get(stage, "No output available") for stage in st.session_state.completed_tasks])}

            CONTEXTUAL CONSIDERATIONS:
            - Challenge Context: {st.session_state.project_input['context']}
            - Key Constraints: {str(st.session_state.project_input['constraints'])}
            
            EXPECTED DELIVERABLES:
            1. Comprehensive Decision Journey Map
            2. Agent Decision Rationale Analysis
            3. Cross-Stage Impact Assessment
            4. Manager Coordination Report
            5. Context Flow Documentation""",
            "expected_output": """Provide a structured analysis including:

            1. Decision Journey Map:
                - Chronological mapping of key decisions
                - Decision points and their rationale
                - Alternative paths considered
                - Impact assessment of each decision

            2. Agent Decision Analysis:
                - Each agent's decision-making process
                - Rationale documentation
                - Methodology explanation
                - Impact evaluation

            3. Cross-Stage Analysis:
                - Information flow between stages
                - Decision dependencies
                - Ripple effects
                - Evolution of understanding

            4. Manager Coordination Report:
                - Task distribution strategy
                - Communication protocols
                - Conflict resolution methods
                - Resource allocation decisions

            5. Context Integration Analysis:
                - Knowledge transfer between stages
                - Context utilization documentation
                - Insight application tracking
                - Decision influence mapping""",
            "agent": st.session_state.crew.reporting_agent,
            "show_file_upload": False
        },
        "report": {
            "name": "Final Report",
            "description": f"""Create a comprehensive markdown report documenting the entire
            design thinking process. Include insights, outcomes, and research methodology from each stage.
            
            RESEARCH METHODOLOGY AND SOURCES:
            {prepare_research_summary()}
            
            Please incorporate relevant research queries and citations throughout the report where appropriate.""",
            "expected_output": """A detailed markdown report including:
            1. Executive Summary
            2. Research Methodology
                - Search queries used in each stage
                - Key information sources
                - Research timeline
            3. User Research Findings
            4. Problem Definition
            5. Ideation Process and Solutions
            6. Prototype Details
            7. Testing Results and Recommendations
            8. Next Steps
            9. References and Citations
                - Comprehensive list of all sources
                - Search queries that led to key insights
                - Source credibility assessment
            """,
            "agent": st.session_state.crew.reporting_agent,
            "show_file_upload": False
        },
        "manager_briefing_task": {
            "name": "Manager Briefing",
            "description": f"""COMPREHENSIVE COORDINATION AND OVERSIGHT REPORT

            DESIGN CHALLENGE PARAMETERS:
            Challenge: {st.session_state.project_input['challenge']}
            Context: {st.session_state.project_input['context']}
            Constraints: {str(st.session_state.project_input['constraints'])}
            Main POV: {st.session_state.main_pov if 'main_pov' in st.session_state else "Not yet defined"}

            COORDINATION REQUIREMENTS:

            1. Stage Coordination Strategy
                - Document how each stage was coordinated
                - Explain context passing between stages
                - Detail agent collaboration protocols
                - Track information flow and dependencies

            2. Resource Management
                - Document how resources were allocated
                - Explain task prioritization decisions
                - Detail workload distribution
                - Track efficiency optimization methods

            3. Quality Control Measures
                - Document review and validation processes
                - Explain error correction protocols
                - Detail quality assurance methods
                - Track performance optimization efforts

            4. Communication Framework
                - Document inter-agent communication protocols
                - Explain information sharing methods
                - Detail conflict resolution approaches
                - Track collaboration effectiveness

            5. Process Optimization
                - Document workflow improvements
                - Explain efficiency enhancements
                - Detail bottleneck resolution
                - Track overall process effectiveness

            PROVIDE DETAILED DOCUMENTATION OF:
            - How each stage built upon previous stages
            - How context was maintained and transferred
            - How agent collaboration was facilitated
            - How challenges were addressed and resolved
            """,
            "expected_output": """Provide a comprehensive management report including:

            1. Coordination Strategy:
                - Stage-by-stage coordination approach
                - Context management methods
                - Agent collaboration protocols
                - Information flow optimization

            2. Resource Allocation:
                - Task distribution methodology
                - Priority management approach
                - Efficiency optimization methods
                - Performance tracking metrics

            3. Quality Assurance:
                - Validation processes
                - Error prevention methods
                - Quality control protocols
                - Performance optimization results

            4. Communication Management:
                - Inter-agent communication framework
                - Information sharing protocols
                - Conflict resolution methods
                - Collaboration effectiveness metrics

            5. Process Optimization:
                - Workflow improvements implemented
                - Efficiency enhancements achieved
                - Challenge resolution methods
                - Overall process effectiveness analysis""",
            "agent": st.session_state.crew.manager_agent
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
            "deepseek": "",
            "claude": "",
            "groq": ""
        }
        
    if 'main_pov' not in st.session_state:
        st.session_state.main_pov = ""
        
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
    
    if 'search_queries' not in st.session_state:
        st.session_state.search_queries = {
            'empathize': [],
            'define': [],
            'ideate': [],
            'prototype': [],
            'test': [],
            'decisions': [],
            'report': []
        }
    
    if 'citations' not in st.session_state:
        st.session_state.citations = {}

# Functions for the Streamlit UI
def setup_api_keys():
    st.sidebar.header("API Keys")
    
    gemini_key = st.sidebar.text_input("Gemini API Key", value=st.session_state.api_keys.get("gemini", ""), type="password")
    serper_key = st.sidebar.text_input("Serper API Key", value=st.session_state.api_keys.get("serper", ""), type="password")
    claude_key = st.sidebar.text_input("Claude API Key", value=st.session_state.api_keys.get("claude", ""), type="password")
    groq_key = st.sidebar.text_input("Groq API Key", value=st.session_state.api_keys.get("groq", ""), type="password")
    openai_key = st.sidebar.text_input("OpenAI API Key (Optional)", value=st.session_state.api_keys.get("openai", ""), type="password")
    deepseek_key = st.sidebar.text_input("DeepSeek API Key (Optional)", value=st.session_state.api_keys.get("deepseek", ""), type="password")
    
    st.session_state.api_keys = {
        "gemini": gemini_key,
        "serper": serper_key,
        "claude": claude_key,
        "groq": groq_key,
        "openai": openai_key,
        "deepseek": deepseek_key
    }
    
    # Model selection
    model_options = {
        "Gemini 2.0 Flash": "gemini/gemini-2.0-flash",     #-thinking-exp-01-21
        "Claude 3.7 Sonnet": "anthropic/claude-3-7-sonnet-20250219",
        "Groq DeepSeek Llama 70B": "groq/deepseek-r1-distill-qwen-32b",  #deepseek-r1-distill-llama-70b",
        "DeepSeek Reasoner": "openrouter/nvidia/llama-3.1-nemotron-70b-instruct", 
        "OpenAI o3-mini": "openai/o3-mini"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select LLM Model",
        options=list(model_options.keys()),
        index=0
    )
    
    st.session_state.model_name = model_options[selected_model]
    
    # Add PDF upload for design thinking guide
    st.sidebar.header("Design Thinking Guide")
    uploaded_guide = st.sidebar.file_uploader(
        "Upload Design Thinking Guide (PDF)",
        type=["pdf"],
        help="Upload a PDF containing design thinking methodology and best practices"
    )
    
    if uploaded_guide:
        try:
            # Create a temporary file to store the PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_guide.getvalue())
                guide_path = tmp_file.name
            
            # Initialize crew with the guide
            if gemini_key and serper_key:
                try:
                    st.session_state.crew = DesignThinkingCrew(
                        api_keys=st.session_state.api_keys,
                        model_name=st.session_state.model_name
                    )
                    # Update the RAG tool with the uploaded guide
                    st.session_state.crew.design_thinking_knowledge = RagTool(
                        config=dict(
                            llm=dict(
                                provider="google",
                                config=dict(
                                    model=st.session_state.model_name,
                                    api_key=gemini_key
                                ),
                            ),
                            embedder=dict(
                                provider="google",
                                config=dict(
                                    model="models/embedding-001",
                                    task_type="retrieval_document",
                                ),
                            ),
                        ),
                        files=[guide_path],
                        name="Design Thinking Guide",
                        description="A comprehensive guide for design thinking methodology and best practices"
                    )
                    st.sidebar.success("Design Thinking Crew initialized with methodology guide!")
                except Exception as e:
                    st.sidebar.error(f"Error initializing crew with guide: {e}")
        except Exception as e:
            st.sidebar.error(f"Error processing design thinking guide: {e}")

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
        st.subheader("Provide Challenge Details")
        challenge = st.text_area("Challenge Statement", placeholder="Enter the design challenge statement...")
        st.subheader("Provide Context Details")
        context = st.text_area("Context", placeholder="Provide background information about the problem...")
        st.subheader("Provide Constraint Details")
        constraints = st.text_area("Constraints", placeholder="List the constraints, examples:\n- Must be low cost, or cannot use electricity, or needs to meet specific laws")


        if st.button("Set Challenge"):
            constraints_list = [c.strip() for c in constraints.split("\n") if c.strip()]
            st.session_state.project_input = {
                "challenge": challenge,
                "context": context,
                "constraints": constraints_list if constraints_list else "No specific constraints provided."
            }
            st.success("Challenge set successfully!")
            
            # Add tab switch button and script
            if st.button("Start Design Thinking Process ➡️", key="start_process_button"):
                # Switch to Design Thinking Process tab (index 1)
                html("""
                <script>
                (() => {
                    let button = [...window.parent.document.querySelectorAll("button")].filter(button => {
                        return button.innerText.includes("Start Design Thinking Process")
                    })[0];
                    if(button) {
                        button.onclick = () => {
                            var tabGroup = window.parent.document.getElementsByClassName("stTabs")[0];
                            const tabButton = [...tabGroup.querySelectorAll("button")].filter(button => {
                                return button.innerText.includes("Design Thinking Process")
                            })[0];
                            if(tabButton) {
                                tabButton.click();
                            } else {
                                console.log("Design Thinking Process tab button not found");
                            }
                        }
                    } else {
                        console.log("Start process button not found");
                    }
                })();
                </script>
                """, height=0)
                st.rerun()
            
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
                        
                        # Add tab switch button and script
                        if st.button("Start Design Thinking Process ➡️", key="start_process_button"):
                            html("""
                            <script>
                            (() => {
                                const tabGroups = window.parent.document.getElementsByClassName("stTabs");
                                const mainTabs = tabGroups[0];
                                if (mainTabs) {
                                    const designThinkingTab = mainTabs.querySelectorAll("button")[1];
                                    if (designThinkingTab) {
                                        designThinkingTab.click();
                                    }
                                }
                            })();
                            </script>
                            """, height=0)
                            st.rerun()
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
                        
                        # Define manager briefing task directly
                        manager_briefing_task = Task(
                            description=f"""IMPORTANT: Review and understand this context.
                            
                            CONTEXT: {context}
                            
                            Your job is to help identify the domain and potential challenges in this context.
                            """,
                            expected_output="Confirmation of understanding the context",
                            agent=st.session_state.crew.manager_agent
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
                            
                        # Generate full challenge with the domain we've determined
                        st.session_state.project_input = st.session_state.crew.generate_challenge(
                            domain=domain_section,
                            context=context,
                            constraints=constraints_list if constraints_list else None
                        )
                        st.success("Challenge extracted successfully!")
                        
                        # Add tab switch button and script
                        if st.button("Start Design Thinking Process ➡️", key="start_process_button"):
                            html("""
                            <script>
                            (() => {
                                const tabGroups = window.parent.document.getElementsByClassName("stTabs");
                                const mainTabs = tabGroups[0];
                                if (mainTabs) {
                                    const designThinkingTab = mainTabs.querySelectorAll("button")[1];
                                    if (designThinkingTab) {
                                        designThinkingTab.click();
                                    }
                                }
                            })();
                            </script>
                            """, height=0)
                            st.rerun()

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
    
    # We'll remove the chat_input from here since it can't be used inside tabs
    # We'll set a session state variable to indicate which agent we're chatting with
    st.session_state.current_chat_stage = stage_name
    st.session_state.current_chat_agent = agent_role

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

def display_research_analytics():
    st.subheader("Research Analytics")
    
    # Create metrics section
    col1, col2, col3 = st.columns(3)
    with col1:
        total_queries = sum(len(queries) for queries in st.session_state.search_queries.values())
        st.metric("Total Search Queries", total_queries)
    
    with col2:
        total_sources = len({citation['link'] 
                           for citations in st.session_state.citations.values() 
                           for citation in citations})
        st.metric("Unique Sources", total_sources)
    
    with col3:
        stages_with_research = sum(1 for queries in st.session_state.search_queries.values() if queries)
        st.metric("Stages with Research", f"{stages_with_research}/5")

    # Create query analysis by stage
    st.subheader("Research Distribution by Stage")
    stage_data = {stage: len(queries) for stage, queries in st.session_state.search_queries.items()}
    st.bar_chart(stage_data)

    # Show top domains referenced
    st.subheader("Top Referenced Domains")
    domain_counts = {}
    for citations in st.session_state.citations.values():
        for citation in citations:
            from urllib.parse import urlparse
            domain = urlparse(citation['link']).netloc
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    # Sort and display top domains
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for domain, count in sorted_domains:
        st.write(f"- {domain}: {count} references")

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

    # Handle tab switching from navigation
    if 'switch_tab' in st.session_state:
        tab_index = st.session_state.switch_tab
        del st.session_state.switch_tab  # Clear the switch flag
        st.session_state.active_tab_index = tab_index
        
    current_index = task_order.index(st.session_state.current_stage)

    # Update active tab index to match current stage
    st.session_state.active_tab_index = current_index
    
    # Handle content for all tabs so it persists
    for tab_index, stage in enumerate(task_order):
        with tabs[tab_index]:
            # Add stage indicator
            st.info(f"Currently in: {task_definitions[stage]['name']} stage", icon="📍")
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
                    with st.status("Processing PDF files..."):
                        pdf_contents = []
                        for uploaded_file in uploaded_files:
                            st.write(f"Processing: {uploaded_file.name}")
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            # Extract text from PDF
                            pdf_text = extract_text_from_pdf(tmp_file_path)
                            if not pdf_text.startswith('[Error'):
                                pdf_contents.append(pdf_text)
                                st.write(f"✅ Successfully processed {uploaded_file.name}")
                            else:
                                st.write(f"❌ Failed to process {uploaded_file.name}")
                            
                            # Delete temporary file
                            os.unlink(tmp_file_path)
                        
                        # Store PDF contents for this stage
                        st.session_state.uploaded_pdfs[stage] = pdf_contents
                        
                        st.success(f"{len(pdf_contents)} PDF files processed successfully!")
            
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
                
                # Get solutions from prototype stage if available
                prototype_solutions = []
                
                # First try to get from selected solutions
                if "selected_solutions" in st.session_state and "prototype" in st.session_state.selected_solutions:
                    prototype_solutions = st.session_state.selected_solutions["prototype"]
                
                # If no selected solutions, try to parse from prototype output
                if not prototype_solutions:
                    prototype_output = st.session_state.task_outputs["prototype"]
                    
                    # Try to extract prototyped solutions using multiple methods
                    import re
                    
                    # Look for solution headers with different patterns
                    patterns = [
                        r'(?:Prototype\s*\d+:?\s*)(.*?)(?=Prototype|$)',  # Prototype headers
                        r'(?:Solution\s*\d+:?\s*)(.*?)(?=Solution|$)',    # Solution headers
                        r'(?:\d+\.\s*)(.*?)(?=\d+\.|$)',                 # Numbered lists
                        r'(?:\*\s*)(.*?)(?=\*|$)'                        # Bullet points
                    ]
                    
                    for pattern in patterns:
                        found_solutions = re.findall(pattern, prototype_output, re.DOTALL)
                        if found_solutions:
                            # Clean up solutions
                            cleaned_solutions = []
                            for solution in found_solutions:
                                # Clean up the solution text
                                cleaned = solution.strip()
                                # Remove any markdown formatting
                                cleaned = re.sub(r'[#*_`]', '', cleaned)
                                # Limit length and add if not empty
                                if cleaned and len(cleaned) > 5:  # Minimum length check
                                    if len(cleaned) > 200:
                                        cleaned = cleaned[:197] + "..."
                                    cleaned_solutions.append(cleaned)
                            
                            if cleaned_solutions:
                                prototype_solutions = cleaned_solutions
                                break
                
                # If still no solutions found, provide default
                if not prototype_solutions:
                    st.warning("No prototype solutions could be automatically extracted. Please enter them manually.")
                    manual_input = st.text_area("Enter prototype solutions (one per line):")
                    if manual_input:
                        prototype_solutions = [s.strip() for s in manual_input.split('\n') if s.strip()]
                
                # Display solutions for selection
                if prototype_solutions:
                    st.write("Select prototypes to test:")
                    selected_test_solutions = []
                    for i, solution in enumerate(prototype_solutions):
                        if st.checkbox(f"Test: {solution}", key=f"test_solution_{i}", value=True):
                            selected_test_solutions.append(solution)
                    
                    # Store selected solutions
                    if "selected_test_solutions" not in st.session_state:
                        st.session_state.selected_test_solutions = {}
                    st.session_state.selected_test_solutions[stage] = selected_test_solutions
                    
                    # Add selected solutions to task description
                    if selected_test_solutions:
                        task_description += "\n\nSELECTED PROTOTYPES TO TEST:\n"
                        for i, sol in enumerate(selected_test_solutions, 1):
                            task_description += f"{i}. {sol}\n"
                        
                        # Make sure this is prominently displayed
                        st.info(f"You've selected {len(selected_test_solutions)} prototypes for testing")
                else:
                    st.error("No prototypes available for testing. Please complete the prototype stage first.")
                
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
                    output_text = st.session_state.task_outputs.get(stage, "No output available")
                    cleaned_output = output_text.replace("```markdown", "").replace("```", "")
                    st.markdown(cleaned_output)
                    
                    try:
                        # Add PDF download button
                        pdf_bytes = create_pdf_from_markdown(cleaned_output, f"{task_definitions[stage]['name']} Output")
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"{stage}_report.pdf",
                            mime="application/pdf",
                            key=f"pdf_download_{stage}"
                        )
                    except Exception as e:
                        st.error(f"Could not generate PDF. Error: {str(e)}")
                
                    # Show research queries for this stage
                    if st.session_state.search_queries.get(stage):
                        with st.expander("View Research Queries", expanded=False):
                            st.markdown("### Search Queries Used in This Stage")
                            for query_info in st.session_state.search_queries[stage]:
                                st.markdown(f"""
                                - Time: {query_info['timestamp']}
                                Query: `{query_info['query']}`
                                """)
                                
                                # Show citations for this query
                                if query_info['query'] in st.session_state.citations:
                                    st.markdown("  Sources found:")
                                    for citation in st.session_state.citations[query_info['query']]:
                                        st.markdown(f"""
                                        > - [{citation['title']}]({citation['link']})
                                        >   _{citation['snippet']}_
                                        """)

                # Show Main POV input after Empathize stage completion
                if stage == "empathize":
                    st.subheader("Define Main Point of View")
                    st.info("Now that we've gathered empathy insights, please define the main point of view that will guide the rest of the process.")
                    main_pov = st.text_area(
                        "Main Point of View",
                        value=st.session_state.main_pov,
                        help="This perspective will be carried through all subsequent stages",
                        key="main_pov_after_empathize"
                    )
                    if main_pov != st.session_state.main_pov:
                        st.session_state.main_pov = main_pov
                        st.success("Main POV updated!")


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
                if stage == "empathize":
                    st.markdown("""
                    ### About the Empathy Research Agent
                    
                    Here is the Empathy Research Agent - a world-class ethnoresearcher with a primary goal of uncovering human needs, 
                    pain points, and motivations to help establish a foundation for user-centered research.

                    This Agent derives an empathy map from its internal, established databases or from uploaded reference PDF/documents. 
                    This could include:
                    - Interview transcripts
                    - Market research
                    - Consumer study notes
                    - User feedback
                    - Behavioral analytics
                    - Observational data

                    The Agent analyzes this information to create comprehensive empathy maps that capture the full spectrum of user experiences.
                    
                    ---
                    """)
                    
                        
                elif stage == "define":
                    st.markdown("""
                    ### About the Problem Definition Specialist

                    Here is the Problem Definition Specialist agent - a master analyst trained to transform complex research data 
                    into concise problem statements which reveal core user needs, challenge assumptions, and provide a solid 
                    foundation for breakthrough innovation.

                    This agent will leverage learnings from the Empathy Agent as well as any pertinent files you choose to upload. 
                    These could include:
                    - Market research documents
                    - Predetermined personas of interest
                    - User research findings
                    - Industry reports
                    - Stakeholder requirements

                    You can also proceed without uploading additional files - this agent will effectively utilize the insights 
                    gathered from your work with the Empathy agent to define the problem space.

                    The agent excels at synthesizing information to create actionable problem statements that capture the 
                    essence of user needs while maintaining solution neutrality.
                    
                    ---
                    """)
                elif stage == "ideate":
                    st.markdown("""
                    ### About the Innovation Facilitation Agent

                    Here is the Innovation Facilitation Agent - trained by top innovation hubs worldwide with a focus on orchestrating 
                    breakthrough, radical, and diverse solutions through systemic exploration of possibility white spaces.

                    Just as with the other agents, you can upload:
                    - Supporting ideation documentation
                    - Study information
                    - Market research
                    - Preferred ideation methods
                    
                    Feel free to leave this section empty and have the agent run the ideation session on its own, leveraging 
                    insights from previous stages to generate innovative solutions.
                    
                    ---
                    """)
                elif stage == "prototype":
                    st.markdown("""
                    ### About the Prototyping Specialist Agent

                    Here is the Prototyping Specialist Agent - trained to transform abstract concepts from the ideation session 
                    into tangible and testable experiences. This is done with a focus on developing a Minimal Viable Product, 
                    to ensure the most efficient use of resources are dedicated to validating the most critical assumptions.

                    Though not needed, you can upload:
                    - Existing prototyping plans
                    - Example prototypes
                    - Design specifications
                    - Resource constraints

                    For this agent, you will select the solutions (developed with the Ideate Agent) that you would like to 
                    move forward with creating a prototyping plan for.
                    
                    ---
                    """)
                elif stage == "test":
                    st.markdown("""
                    ### About the User Testing Coordinator Agent

                    Here is the User Testing Coordinator Agent - trained as a renowned methodologist for user testing to design 
                    testing protocols that extract learnings from user interactions, validate critical assumptions, and guide 
                    iterative refinement toward solutions users truly value.

                    This agent is meant to be used iteratively with the testing ideas laid out by your discussion with the 
                    prototyping specialist agent. You can upload relevant information such as:
                    - Preliminary test results
                    - Meeting minute updates from prototyping discussions
                    - Current testing plan drafts that need refinement
                    - User feedback protocols
                    - Testing metrics and KPIs

                    The agent will help design comprehensive testing protocols tailored to your specific prototypes and user needs.
                    
                    ---
                    """)


                # Display the task description and expected output
                st.markdown("### Task Description")
                if stage == "empathize":
                    st.markdown(f"This task is derived from the challenge: ")
                st.markdown(f"Challenge: {st.session_state.project_input['challenge']}")
                st.markdown(f"Context: {st.session_state.project_input['context']}")
                st.markdown(f"Constraints: {st.session_state.project_input['constraints']}")

               
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
                            # Run manager briefing task first to ensure it understands the challenge
                            manager_briefing_task = Task(
                                description=f"""IMPORTANT: Review and understand the design challenge before coordinating other agents.

                                DESIGN CHALLENGE: {st.session_state.project_input['challenge']}
                                
                                CONTEXT: {st.session_state.project_input['context']}
                                
                                CONSTRAINTS: {str(st.session_state.project_input['constraints'])}
                                
                                MAIN POINT OF VIEW: {st.session_state.main_pov if 'main_pov' in st.session_state and st.session_state.main_pov else "No specific POV provided."}
                                
                                Your job is to ensure all agents focus their work specifically on this challenge, context, constraints, and maintain the main point of view.
                                When coordinating their work, continually remind them to refer back to these project parameters.
                                """,
                                expected_output="Confirmation of understanding the design challenge and plan for coordination",
                                agent=st.session_state.crew.manager_agent
                            )
                            
                            # Run the briefing task first
                            briefing_crew = Crew(
                                agents=[st.session_state.crew.manager_agent],
                                tasks=[manager_briefing_task],
                                verbose=True
                            )
                            
                            briefing_result = briefing_crew.kickoff(
                                inputs={
                                    "project_input": st.session_state.project_input,
                                    "challenge": st.session_state.project_input['challenge'],
                                    "context": st.session_state.project_input['context'],
                                    "constraints": st.session_state.project_input['constraints']
                                }
                            )
                            
                            # Set context based on completed tasks
                            context_tasks = []
                            
                            # Define context based on design thinking flow
                            if stage == "define" and "empathize" in st.session_state.completed_tasks:
                                context_tasks.append({
                                    "stage": "empathize",
                                    "output": st.session_state.task_outputs["empathize"]
                                })
                                
                            elif stage == "ideate":
                                # Add Define output
                                if "define" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "define",
                                        "output": st.session_state.task_outputs["define"]
                                    })
                                # Add Empathize output
                                if "empathize" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "empathize",
                                        "output": st.session_state.task_outputs["empathize"]
                                    })

                            elif stage == "prototype":
                                # Add Ideate output
                                if "ideate" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "ideate",
                                        "output": st.session_state.task_outputs["ideate"]
                                    })
                                # Add Define output
                                if "define" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "define",
                                        "output": st.session_state.task_outputs["define"]
                                    })
                                # Add Empathize output
                                if "empathize" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "empathize",
                                        "output": st.session_state.task_outputs["empathize"]
                                    })

                            elif stage == "test":
                                # Add Prototype output
                                if "prototype" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "prototype",
                                        "output": st.session_state.task_outputs["prototype"]
                                    })
                                # Add Ideate output
                                if "ideate" in st.session_state.completed_tasks:
                                    context_tasks.append({
                                        "stage": "ideate",
                                        "output": st.session_state.task_outputs["ideate"]
                                    })
                                # Add previous outputs
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
                                
                            elif stage == "decisions":
                                # Add all previous tasks as context
                                for name in ["empathize", "define", "ideate", "prototype", "test"]:
                                    if name in st.session_state.completed_tasks:
                                        context_tasks.append({
                                            "stage": name,
                                            "output": st.session_state.task_outputs[name]
                                        })
                                
                            elif stage == "report":
                                # First add all completed tasks
                                for name in st.session_state.completed_tasks:
                                    if name in st.session_state.task_outputs:
                                        context_tasks.append({
                                            "stage": name,
                                            "output": st.session_state.task_outputs[name]
                                        })
                                
                                # Add research methodology and citations
                                research_summary = "\n\n## Research Methodology and Sources\n\n"
                                
                                # Add queries by stage
                                for stage_name, queries in st.session_state.search_queries.items():
                                    if queries:
                                        research_summary += f"\n### {stage_name.capitalize()} Stage Research\n"
                                        for query_info in queries:
                                            research_summary += f"- Search Query: `{query_info['query']}`\n"
                                            # Add citations for this query
                                            if query_info['query'] in st.session_state.citations:
                                                research_summary += "  Sources:\n"
                                                for citation in st.session_state.citations[query_info['query']]:
                                                    research_summary += f"  - [{citation['title']}]({citation['link']})\n"
                                                    research_summary += f"    - {citation['snippet']}\n"
                                
                                # Add research summary to context
                                context_tasks.append({
                                    "stage": "research_methodology",
                                    "output": research_summary
                                })
                            

                            # Add the context from previous tasks to the task description
                            if context_tasks and len(context_tasks) > 0:
                                task_description += "\n\nPREVIOUS STAGES OUTPUTS:\n"
                                for context in context_tasks:
                                    task_description += f"\n## {context['stage'].upper()} STAGE OUTPUT:\n{context['output']}\n"
                                
                            # Add Main POV to context information for all stages
                            if 'main_pov' in st.session_state and st.session_state.main_pov:
                                task_description += f"\n\nMAIN POINT OF VIEW: {st.session_state.main_pov}\n"
                                task_description += "This POV should be the lens through which you approach this stage of the design thinking process."
                            
                            # Extract any manager logs/interactions for process logs
                            manager_logs = []
                            
                            # Add logging/validation
                            st.session_state.process_logs.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "stage": stage,
                                "action": f"Executed {current_task_def['name']} task",
                                "context_passed": [ctx["stage"] for ctx in context_tasks],
                                "manager_notes": manager_logs
                            })

                            # Optional: Add visual confirmation
                            if context_tasks:
                                st.write("Previous stage outputs included:")
                                for ctx in context_tasks:
                                    st.write(f"- {ctx['stage'].capitalize()} stage output")

                            # Add PDF content to the task description
                            pdf_contents = st.session_state.uploaded_pdfs.get(stage, [])
                            if pdf_contents and len(pdf_contents) > 0:
                                task_description += "\n\nREFERENCE PDF DOCUMENTS:\n"
                                for i, content in enumerate(pdf_contents):
                                    # Limit content length to avoid token issues
                                    truncated_content = content[:8000] + "..." if len(content) > 8000 else content
                                    task_description += f"\nPDF DOCUMENT {i+1}:\n{truncated_content}\n\n"
                            
                            # Ensure the design challenge, context, constraints, and Main POV are prominently included
                            if 'main_pov' in st.session_state and st.session_state.main_pov:
                                task_description = f"""IMPORTANT: Focus on this specific design challenge, context, and constraints.

                            DESIGN CHALLENGE: {st.session_state.project_input['challenge']}

                            CONTEXT: {st.session_state.project_input['context']}

                            CONSTRAINTS: {str(st.session_state.project_input['constraints'])}

                            MAIN POINT OF VIEW: {st.session_state.main_pov}
                            This POV should drive your thinking throughout this task.

                            TASK INSTRUCTIONS:
                            {task_description}
                            """
                            else:
                                task_description = f"""IMPORTANT: Focus on this specific design challenge, context, and constraints.

                            DESIGN CHALLENGE: {st.session_state.project_input['challenge']}

                            CONTEXT: {st.session_state.project_input['context']}

                            CONSTRAINTS: {str(st.session_state.project_input['constraints'])}

                            TASK INSTRUCTIONS:
                            {task_description}
                            """
                            
                            # Special handling for prototype stage - make sure all selected solutions are emphasized
                            if stage == "prototype" and "selected_solutions" in st.session_state and "prototype" in st.session_state.selected_solutions:
                                solutions = st.session_state.selected_solutions.get("prototype", [])
                                if solutions:
                                    solution_text = "\n".join(f"Solution {i+1}: {sol}" for i, sol in enumerate(solutions))
                                else:
                                    solution_text = "No solutions selected"
                                task_description += f"""

                                \n\nIMPORTANT: You MUST create detailed prototype descriptions for ALL {len(solutions)} 
                                selected solutions listed below. Each solution must be given equal attention and detail.
                                
                                SELECTED SOLUTIONS TO PROTOTYPE:
                                {chr(10).join([f"{i+1}. {sol}" for i, sol in enumerate(solutions)])}
                                
                                For each solution above, provide:
                                - Detailed prototype specifications
                                - Core features
                                - Development approach
                                - Implementation considerations
                                
                                DO NOT focus on just one solution - you must prototype ALL {len(solutions)} selected solutions.

                                Selected Solutions:
                                 {solution_text}
                                """
                            
                            # Special handling for test stage - make sure all selected prototypes are emphasized
                            if stage == "test" and "selected_test_solutions" in st.session_state and stage in st.session_state.selected_test_solutions:
                                selected_tests = st.session_state.selected_test_solutions[stage]
                                if selected_tests:
                                    task_description += f"""
                                    \n\nIMPORTANT: You MUST create comprehensive testing protocols for ALL {len(selected_tests)} 
                                    selected prototypes listed below. Each prototype must have its own detailed testing approach.
                                    
                                    SELECTED PROTOTYPES TO TEST:
                                    {chr(10).join([f"{i+1}. {test}" for i, test in enumerate(selected_tests)])}
                                    
                                    For each prototype above, provide:
                                    - Specific testing methodology
                                    - User feedback collection approach
                                    - Success metrics
                                    - Acceptance criteria
                                    
                                    DO NOT focus on just one prototype - you must test ALL {len(selected_tests)} selected prototypes.
                                    """
                            
                            # Now create the task with the updated description
                            task = Task(
                                description=sanitize_task_description(task_description),  # Add sanitize_task_description here
                                expected_output=current_task_def["expected_output"],
                                agent=current_task_def["agent"]
                            )
                            
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
                                        "pdf_contents": pdf_contents
                                    }
                                )
                                st.write("Task completed successfully!")
                            

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
    
                
            
    # Get the task definitions and define task order
    task_definitions = get_task_definitions(st.session_state)
    task_order = list(task_definitions.keys())

    # Create separate chat containers for each stage
    chat_containers = {}
    for tab_index, stage in enumerate(task_order):
        chat_containers[stage] = st.container(key=f"chat_container_{stage}_{tab_index}")

    # Later in the chat logic section:
    if 'current_chat_stage' in st.session_state and 'current_chat_agent' in st.session_state:
        stage_name = st.session_state.current_chat_stage
        agent_role = st.session_state.current_chat_agent
        
        # Get the appropriate chat container for this stage
        chat_container = chat_containers.get(stage_name)
        
        if chat_container and stage_name in st.session_state.completed_tasks:
            with chat_container:
                # Create a unique hash that includes the tab iteration index
                import hashlib
                current_tab = st.session_state.active_tab_index
                tab_iteration = task_order.index(stage_name)
                
                # Include more components in the hash for increased uniqueness
                hash_input = f"{stage_name}_{current_tab}_{agent_role}_{tab_iteration}"
                unique_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
                
                # Display who you're chatting with
                st.write(f"💬 Chat with {agent_role}")
                
                # Create the chat input with the unique key
                user_input = st.chat_input(
                    f"Ask a question or type 'regenerate' with instructions...",
                    key=f"chat_input_{stage_name}_{tab_iteration}_{unique_hash}"
                )
                
                # Process user input
                if user_input:
                    # Add user message to chat history
                    if stage_name not in st.session_state.chat_history:
                        st.session_state.chat_history[stage_name] = []
                        
                    st.session_state.chat_history[stage_name].append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Check if regeneration request
                    if "redo" in user_input.lower() or "regenerate" in user_input.lower():
                        # Process regeneration request
                        # Extract instructions
                        instructions = user_input.lower()
                        if "regenerate" in instructions:
                            instructions = instructions.split("regenerate", 1)[1].strip()
                        elif "redo" in instructions:
                            instructions = instructions.split("redo", 1)[1].strip()
                        
                        # Add agent response
                        st.session_state.chat_history[stage_name].append({
                            "role": "assistant",
                            "content": f"I'll regenerate the content with your instructions: '{instructions}'. Please wait...",
                            "avatar": "🧠"
                        })
                        
                        # Update suggestions
                        if stage_name in st.session_state.stage_suggestions:
                            st.session_state.stage_suggestions[stage_name] += f"\n- {instructions}"
                        else:
                            st.session_state.stage_suggestions[stage_name] = f"- {instructions}"
                        
                        # Record decision
                        if st.session_state.crew:
                            st.session_state.crew.decision_tracker.record_decision(
                                stage=f"{stage_name.capitalize()} Regeneration",
                                decision=f"Regenerating content with new instructions",
                                rationale=f"User requested: {instructions}"
                            )
                        
                        # Reset the task
                        if stage_name in st.session_state.completed_tasks:
                            st.session_state.completed_tasks.pop(stage_name, None)
                        
                        # Reset navigation state before rerunning
                        if 'current_chat_stage' in st.session_state:
                            del st.session_state.current_chat_stage
                        if 'current_chat_agent' in st.session_state:
                            del st.session_state.current_chat_agent
                        
                        # Move to next stage if it exists
                        current_index = task_order.index(stage_name)
                        if current_index < len(task_order) - 1:
                            next_stage = task_order[current_index + 1]
                            st.session_state.current_stage = next_stage
                            st.session_state.active_tab_index = current_index + 1
                        
                        st.rerun()
                    else:
                        # Handle regular question
                        task_output = st.session_state.task_outputs.get(stage_name, "No output generated yet.")
                        
                        # Create prompt
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
                                # Call LLM
                                response = st.session_state.crew.llm.call(prompt)
                                agent_response = response
                            except Exception as e:
                                agent_response = f"I'm sorry, I had trouble retrieving that information. Error: {str(e)}"
                        else:
                            agent_response = "I'm sorry, I can't access my previous work right now."
                        
                        # Add to chat history
                        st.session_state.chat_history[stage_name].append({
                            "role": "assistant",
                            "content": agent_response,
                            "avatar": "🧠"
                        })
                        
                        # Reset navigation state before rerunning
                        if 'current_chat_stage' in st.session_state:
                            del st.session_state.current_chat_stage
                        if 'current_chat_agent' in st.session_state:
                            del st.session_state.current_chat_agent
                        
                        # Move to next stage if it exists
                        current_index = task_order.index(stage_name)
                        if current_index < len(task_order) - 1:
                            next_stage = task_order[current_index + 1]
                            st.session_state.current_stage = next_stage
                            st.session_state.active_tab_index = current_index + 1
                        
                        st.rerun()

    # Navigation controls
    st.subheader("Navigation")

    def run_stage(stage_name, task_def):
        """Run a single stage and handle errors appropriately"""
        try:
            # Get PDF contents for this stage
            pdf_contents = st.session_state.uploaded_pdfs.get(stage_name, [])
            
            # Get context from previous stages
            context_tasks = []
            for prev_stage in task_order[:task_order.index(stage_name)]:
                if prev_stage in st.session_state.completed_tasks:
                    context_tasks.append({
                        "stage": prev_stage,
                        "output": st.session_state.task_outputs[prev_stage]
                    })
            
            # Create task description with all context
            task_description = task_def["description"]
            
            # Add context from previous stages
            if context_tasks:
                task_description += "\n\nPREVIOUS STAGES OUTPUT:\n"
                for ctx in context_tasks:
                    task_description += f"\n## {ctx['stage'].upper()} STAGE OUTPUT:\n{ctx['output']}\n"
            
            # Create the task
            task = Task(
                description=task_description,
                expected_output=task_def["expected_output"],
                agent=task_def["agent"]
            )
            
            # Create crew for this task
            temp_crew = Crew(
                agents=[task_def["agent"]],
                tasks=[task],
                verbose=True
            )
            
            # Run the task with error handling
            try:
                result = temp_crew.kickoff(
                    inputs={
                        "project_input": st.session_state.project_input,
                        "challenge": st.session_state.project_input['challenge'],
                        "context": st.session_state.project_input['context'],
                        "constraints": st.session_state.project_input['constraints'],
                        "context_tasks": context_tasks,
                        "pdf_contents": pdf_contents
                    }
                )
                
                # Validate result
                if not result or not result.raw:
                    raise ValueError("Empty response from LLM")
                    
                return result
                
            except Exception as e:
                # Handle specific LLM errors
                if "Invalid response from LLM" in str(e) or "Empty response" in str(e):
                    # Retry with backup prompt
                    backup_task = Task(
                        description=f"""BACKUP ATTEMPT - Previous attempt failed. Please try again with this simplified prompt:
                        
                        CHALLENGE: {st.session_state.project_input['challenge']}
                        
                        TASK: {task_description[:500]}... (truncated)
                        
                        Please provide a response for this {stage_name} stage, focusing on the core requirements.
                        """,
                        expected_output=task_def["expected_output"],
                        agent=task_def["agent"]
                    )
                    
                    result = temp_crew.kickoff(
                        inputs={
                            "project_input": st.session_state.project_input,
                            "challenge": st.session_state.project_input['challenge']
                        }
                    )
                    
                    if result and result.raw:
                        return result
                    else:
                        raise ValueError(f"Failed to get valid response from LLM after retry for {stage_name} stage")
                else:
                    raise e
                    
        except Exception as e:
            st.error(f"Error running {task_def['name']}: {str(e)}")
            return None

    col1, col2, col3 = st.columns([1, 2, 1])

    # Helper function for tab switching
    def add_tab_switch_script(button_text, tab_text):
        html(f"""
        <script>
        (() => {{
            let button = [...window.parent.document.querySelectorAll("button")].filter(button => {{
                return button.innerText.includes("{button_text}")
            }})[0];
            if(button) {{
                button.onclick = () => {{
                    var tabGroup = window.parent.document.getElementsByClassName("stTabs")[1];
                    const tabButton = [...tabGroup.querySelectorAll("button")].filter(button => {{
                        return button.innerText.includes("{tab_text}")
                    }})[0];
                    if(tabButton) {{
                        tabButton.click();
                    }} else {{
                        console.log("tab button {tab_text} not found");
                    }}
                }}
            }} else {{
                console.log("button not found: {button_text}");
            }}
        }})();
        </script>
        """, height=0)

    with col1:
        # Previous button
        if current_index > 0:
            prev_stage = task_order[current_index - 1]
            prev_stage_name = task_definitions[prev_stage]['name']
            if st.button(f"⬅️ Previous ({prev_stage_name})", key=f"prev_stage_button_{current_index}"):
                prev_stage_index = current_index - 1
                st.session_state.current_stage = task_order[prev_stage_index]
                st.session_state.active_tab_index = prev_stage_index
                if 'current_chat_stage' in st.session_state:
                    del st.session_state.current_chat_stage
                if 'current_chat_agent' in st.session_state:
                    del st.session_state.current_chat_agent
                st.toast(f"Switching to {prev_stage_name} tab", icon="⬅️")
            # Add tab switch script for previous button
            add_tab_switch_script(f"Previous ({prev_stage_name})", prev_stage_name)

    with col3:
        # Next button
        if current_index < len(task_order) - 1:
            next_stage_index = current_index + 1
            next_stage = task_order[next_stage_index]
            next_stage_name = task_definitions[next_stage]['name']
            if st.button(f"Next ({next_stage_name}) ➡️", key=f"next_stage_button_{current_index}"):
                st.session_state.current_stage = next_stage
                st.session_state.active_tab_index = next_stage_index
                if 'current_chat_stage' in st.session_state:
                    del st.session_state.current_chat_stage
                if 'current_chat_agent' in st.session_state:
                    del st.session_state.current_chat_agent
                st.toast(f"Switching to {next_stage_name} tab", icon="➡️")
            # Add tab switch script for next button
            add_tab_switch_script(f"Next ({next_stage_name})", next_stage_name)

    
    with col2:
        # Jump to any stage
        jump_options = ["Jump to..."] + [task_definitions[stage]["name"] for stage in task_order]
        jump_to = st.selectbox("", jump_options, index=0, key=f"jump_select_{current_index}")
        
        if jump_to != "Jump to...":
            jump_index = [task_definitions[stage]["name"] for stage in task_order].index(jump_to)
            target_stage = task_order[jump_index]
            target_stage_name = task_definitions[target_stage]['name']
            
            # Check if we need to run previous stages
            previous_stages = task_order[:jump_index]
            missing_stages = [stage for stage in previous_stages if stage not in st.session_state.completed_tasks]
            
            if missing_stages:
                st.warning(f"⚠️ Some previous stages haven't been completed: {', '.join([task_definitions[s]['name'] for s in missing_stages])}")
                if st.button(f"Run previous stages and go to {jump_to}", key=f"run_previous_{current_index}"):
                    # Run missing stages sequentially
                    with st.spinner("Running previous stages..."):
                        for stage in missing_stages:
                            try:
                                result = run_stage(stage, task_definitions[stage])
                                if result:
                                    st.session_state.completed_tasks[stage] = True
                                    st.session_state.task_outputs[stage] = result.raw
                            except Exception as e:
                                st.error(f"Error in {task_definitions[stage]['name']}: {str(e)}")
                                break
                    
                    # Now jump to target stage
                    st.session_state.current_stage = target_stage
                    st.session_state.active_tab_index = jump_index
                    st.rerun()
            else:
                if st.button(f"Go to {jump_to}", key=f"jump_button_{current_index}"):
                    st.session_state.current_stage = target_stage
                    st.session_state.active_tab_index = jump_index
                    if 'current_chat_stage' in st.session_state:
                        del st.session_state.current_chat_stage
                    if 'current_chat_agent' in st.session_state:
                        del st.session_state.current_chat_agent
                    st.toast(f"Switching to {target_stage_name} tab", icon="🔄")
                    st.rerun()



def display_decision_log():
    if st.session_state.crew:
        with st.expander("Decision Log"):
            decision_log = st.session_state.crew.decision_tracker.get_decision_log()
            st.markdown(decision_log)

# Main Streamlit UI
def main():
    st.set_page_config(
        page_title="Design Thinking AI Cluster (DTAAC)",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Setup sidebar with API keys
    setup_api_keys()
    
    # App title
    st.title("🧠 Design Thinking AI Cluster (DTAAC)")
    st.markdown("""Welcome to the Design Thinking AI Agent Cluster (DTAAC). 
                This DTAAC platform uses a series of coordinated agent to act as a coach or assistant through your Design Thinking journey. 
                Here we will work together to make sure you follow the Design Thinking guidelines and produce the best product your customer could ask for. 
                We can start this journey in three ways:""") 
    st.markdown("1. Provide the problem statement directly") 
    st.markdown("2. Extract the problem from context clues")
    st.markdown("3. Generate a problem according to a given domain")
    st.markdown("Each one will give a different prompt to better match your intended design journey")
    
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

      
    if 'switch_tab' not in st.session_state:
        st.session_state.switch_tab = None

if __name__ == "__main__":
    main()    
