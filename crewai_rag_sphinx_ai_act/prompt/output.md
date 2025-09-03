# QA AI Agent Application - AI Act Compliance Documentation

**Application Owner**: to be filled  
**Document Version**: 1.0  
**Reviewers**: to be filled

## Key Links

- [Code Repository](c:\Users\XT144AC\Downloads\qa_flow)
- [Deployment Pipeline]() - to be filled
- [API]() ([Swagger Docs]()) - to be filled
- [Cloud Account]() - to be filled
- [Project Management Board]() - to be filled
- [Application Architecture]() - to be filled

## General Information

*EU AI Act [Article 11](https://artificialintelligenceact.eu/article/11/); [Annex IV](https://artificialintelligenceact.eu/annex/4/) paragraph 1, 2, 3*

**Purpose and Intended Use**:

The QA AI Agent is designed for technology companies that primarily sell smartphones to provide their end-users with two main functionalities:

1. **Web Search Capability**: Perform web searches for general information requests that require internet access and current information retrieval.

2. **Product Information Retrieval**: Answer requests specifically related to smartphones by first attempting to retrieve information from the company's product documentation through a Retrieval-Augmented Generation (RAG) system. If no relevant information is found in the product documentation, the query is automatically routed to the web search functionality.

- **Target Users**: End customers of technology companies, specifically those seeking information about smartphones or general topics requiring web search.
- **Stakeholders**: Technology companies selling smartphones, end customers, customer support teams.
- **Measurable Goals and KPIs**: 
  - Query resolution accuracy rate
  - Response time performance
  - User satisfaction scores
  - System uptime and availability
- **Ethical Implications**: The system includes an integrated ethical analysis component that evaluates queries before processing to ensure compliance with safety and ethical standards.
- **Regulatory Constraints**: Fully compliant with EU AI Act requirements for limited-risk AI systems.
- **Prohibited Uses**: The system explicitly prohibits processing of unethical, illegal, harmful, or inappropriate requests through its integrated ethics checking mechanism.
- **Operational Environment**: The system operates through a flow-based architecture using CrewAI framework, designed for deployment on cloud platforms or on-premises infrastructure supporting Python applications.

## Risk Classification

*Prohibited Risk: EU AI Act Chapter II [Article 5](https://artificialintelligenceact.eu/article/5/)  
High-Risk: EU AI Act Chapter III, Section 1 [Article 6](https://artificialintelligenceact.eu/article/6/), [Article 7](https://artificialintelligenceact.eu/article/7/)  
Limited Risk: Chapter IV [Article 50](https://artificialintelligenceact.eu/article/50/)*

**Classification**: Limited Risk

**Reasoning**: The QA AI Agent is classified as a Limited Risk AI system under Chapter IV, Article 50 of the EU AI Act. This classification is appropriate because:

1. The system primarily functions as an information retrieval and query answering interface
2. It does not operate in sensitive domains such as healthcare, law enforcement, education evaluation, or employment decisions
3. The system does not pose safety risks or threaten fundamental rights
4. It includes transparency mechanisms by clearly identifying AI use to users
5. The system incorporates ethical safeguards but does not manipulate human behavior or exploit vulnerabilities

As a Limited Risk system, it must meet transparency obligations, including clear disclosure of AI use to users.

## Application Functionality

*EU AI Act [Article 11](https://artificialintelligenceact.eu/article/11/) ; [Annex IV](https://artificialintelligenceact.eu/annex/4/), paragraph 1, 2, 3*

**Instructions for use for deployers**: The system requires deployment with proper API keys for OpenAI services and SerpDev for web search functionality. Deployers must ensure secure storage of credentials and configure appropriate access controls. The system should be deployed with monitoring capabilities to track performance and ethical compliance.

**Model Capabilities**:
- Process natural language queries in text format
- Perform ethical analysis of user inputs
- Route queries to appropriate information retrieval systems (RAG or web search)
- Retrieve information from pre-indexed smartphone documentation
- Conduct web searches for general information requests
- Generate comprehensive summaries of retrieved information
- Support conversation flows with decision routing based on query analysis

**Limitations**:
- Limited to text-based queries only
- Cannot process multimedia inputs (audio, video, images)
- Dependent on external APIs for web search functionality
- RAG system limited to pre-indexed smartphone documentation
- Cannot perform real-time calculations or execute code
- Does not support multiple languages simultaneously - language support depends on underlying models

**Input Data Requirements**:
- **Format**: Plain text strings representing user queries
- **Quality Expectations**: Clear, coherent natural language questions or information requests
- **Valid Inputs**: Questions about smartphones, general information search requests, topic-specific queries
- **Invalid Inputs**: Unethical requests, illegal content, personal data requests, multimedia content

**Output Explanation**:
- **Response Format**: Structured text summaries with relevant information
- **File Output**: Results are saved to `output/report.md` file
- **Confidence Measures**: System provides routing decisions and fallback mechanisms when information is not found
- **Uncertainty Handling**: When RAG system cannot find relevant smartphone information, queries are automatically routed to web search

**System Architecture Overview**:
The system implements a flow-based architecture using CrewAI framework with the following key components:

1. **GuideCreatorFlow**: Main orchestration flow managing the entire process
2. **EthicCheckerCrew**: AI agents specialized in ethical analysis of user queries
3. **ManagerChoiceCrew**: Routing agents that determine appropriate information retrieval strategy
4. **RagCrew**: Retrieval-Augmented Generation system for smartphone documentation
5. **WebSearchCrew**: Web search agents using SerpDev API
6. **SummarizationCrew**: Summarization agents for processing and presenting results
7. **Custom Tools**: RAG tool with FAISS vector database for document retrieval

## Models and Datasets

*EU AI Act [Article 11](https://artificialintelligenceact.eu/article/11/); [Annex IV](https://artificialintelligenceact.eu/annex/4/) paragraph 2 (d)*

### Models

| Model | Link to Single Source of Truth | Description of Application Usage |
|-------|--------------------------------|----------------------------------|
| OpenAI GPT Models | to be filled | Used by CrewAI agents for natural language processing, ethical analysis, query routing, and summarization tasks |
| Azure OpenAI Embeddings | to be filled | Used for generating document embeddings in the RAG system for similarity search |

### Datasets

| Dataset | Link to Single Source of Truth | Description of Application Usage |
|---------|--------------------------------|----------------------------------|
| Smartphone Documentation | `rsc/docs/` | Pre-indexed smartphone documentation used by RAG system for product-specific queries |
| FAISS Index | `rsc/rag/index.faiss` | Vector database containing document embeddings for similarity search |

## Deployment

### Infrastructure and Environment Details

**Cloud Setup**: to be filled
- Cloud provider specifications required
- Regional deployment requirements to be defined
- Compute, storage, and database service requirements to be specified

**APIs**:
- CrewAI framework integration endpoints
- OpenAI API integration for language models
- SerpDev API for web search functionality
- Authentication: API key-based authentication for external services
- Expected latency: to be filled
- Scalability requirements: to be filled

## Integration with External Systems

*EU AI Act [Article 11](https://artificialintelligenceact.eu/article/11/) ; [Annex IV](https://artificialintelligenceact.eu/annex/4/) paragraph 1 (b, c, d, g, h), 2 (a)*

**Systems**:
- **OpenAI API**: Language model services for agent-based processing
- **SerpDev API**: Web search functionality integration
- **FAISS Vector Database**: Document similarity search and retrieval
- **Local File System**: Document storage and output file management

**Dependencies**: 
- crewai framework (>=0.165.1,<1.0.0)
- faiss-cpu (>=1.12.0)
- langchain community tools
- pydantic for data validation

**Error-handling mechanisms**: 
- Fallback routing from RAG to web search when no relevant documents found
- Ethics checking prevents processing of inappropriate queries
- Retry mechanisms for failed ethical analysis

## Deployment Plan

**Infrastructure**: to be filled
**Integration Steps**: to be filled
**User Information**: to be filled

## Lifecycle Management

*EU AI Act [Article 11](https://artificialintelligenceact.eu/article/11/); [Annex IV](https://artificialintelligenceact.eu/annex/4/) paragraph 6*

**Metrics**:
- **Application Performance**: Response time, error rate, query processing success rate
- **Model Performance**: Ethical analysis accuracy, routing decision accuracy, summarization quality
- **Infrastructure**: CPU usage, memory consumption, API call rates

**Key Activities**:
- Monitor real-world usage patterns and performance metrics
- Update document indices and knowledge base regularly
- Monitor API dependencies for availability and changes
- Regular review of ethical guidelines and compliance

**Documentation Needs**:
- **Monitoring Logs**: Real-time performance data, API response times, system uptime
- **Incident Reports**: System failures, API outages, ethical violations, resolution procedures
- **Update Logs**: Document index updates, configuration changes
- **Audit Trails**: Complete history of system changes and user interactions

**Maintenance of Change Logs**:
- New features: Additional crew capabilities, enhanced routing logic
- Updates: Model version updates, API integrations, performance improvements
- Deprecated features: Legacy routing methods, obsolete documentation
- Removed features: Discontinued APIs, outdated tools
- Bug fixes: Error handling improvements, response formatting corrections
- Security fixes: API key management, access control updates

### Risk Management System

*EU AI Act [Article 9](https://artificialintelligenceact.eu/article/9/)  
EU AI Act [Article 11](https://artificialintelligenceact.eu/article/11/) ; [Annex IV](https://artificialintelligenceact.eu/annex/4/)*

**Risk Assessment Methodology**: The system employs a multi-layered risk assessment approach incorporating ethical analysis at the input stage, with continuous monitoring of system outputs and performance metrics.

**Identified Risks**:

1. **Inappropriate Content Processing**: Risk of processing unethical, illegal, or harmful queries
2. **Information Accuracy**: Risk of providing outdated or incorrect information from web searches
3. **API Dependencies**: Risk of service disruption due to external API failures
4. **Data Privacy**: Risk of inadvertent processing of personal information in queries

**Potential Harmful Outcomes**: 
- Exposure to inappropriate or harmful content
- Distribution of misinformation or outdated product information
- Service unavailability affecting business operations
- Privacy violations through query processing

**Likelihood and Severity**: 
- Inappropriate content: Low likelihood due to ethics checking, Medium severity
- Information accuracy: Medium likelihood, Medium severity
- API failures: Medium likelihood, High severity for service availability
- Privacy issues: Low likelihood due to design, High severity if occurred

#### Risk Mitigation Measures

**Preventive Measures**:
- Integrated ethical analysis crew that evaluates all queries before processing
- Automated routing system that directs queries to appropriate information sources
- Fallback mechanisms when primary information retrieval fails
- Input validation and sanitization processes

**Protective Measures**:
- System restart mechanism when ethical analysis fails
- Automatic retry capabilities for technical failures
- Output monitoring and quality checks
- Secure API key management and access controls

## Testing and Validation (Accuracy, Robustness, Cybersecurity)

*EU AI Act [Article 15](https://artificialintelligenceact.eu/article/15/)*

**Testing and Validation Procedures (Accuracy)**:

**Performance Metrics**: 
- Query routing accuracy (percentage of correctly routed queries)
- Information retrieval success rate
- Response time metrics
- Ethical analysis precision and recall

**Validation Results**: to be filled

**Measures for Accuracy**: 
- High-quality training data validation for document embeddings
- Regular evaluation of routing decision accuracy
- Continuous monitoring of response quality

### Accuracy Throughout the Lifecycle

**Data Quality and Management**:
- Document preprocessing and validation for RAG system
- Regular updates to smartphone documentation database
- Quality checks for web search result relevance
- Input sanitization and validation processes

**Model Selection and Optimisation**:
- Selection of appropriate language models for different crew tasks
- Optimization of vector similarity search parameters
- Fine-tuning of routing decision thresholds
- Performance validation through cross-validation methods

**Feedback Mechanisms**:
- Real-time error tracking and logging
- User feedback collection on response quality
- Automated quality assessment of summaries

### Robustness

**Robustness Measures**:
- Fallback mechanisms when RAG system fails to find relevant information
- Error handling for API failures and network issues
- Retry mechanisms for transient failures
- Graceful degradation when external services are unavailable

**Scenario-Based Testing**:
- Edge case testing with ambiguous queries
- Stress testing with high query volumes
- Failure simulation for external API dependencies
- Adversarial testing for ethical boundary cases

**Redundancy and Fail-Safes**:
- Automatic routing from RAG to web search when no relevant context found
- Multiple fallback options for information retrieval
- Error recovery and restart mechanisms

**Uncertainty Estimation**:
- Confidence scoring for routing decisions
- Quality assessment of retrieved information
- Uncertainty indicators in system responses

### Cybersecurity

*EU AI Act [Article 11](https://artificialintelligenceact.eu/article/11/); [Annex IV](https://artificialintelligenceact.eu/annex/4/) paragraph 2 (h)*

**Data Security**: 
- Secure storage of API keys and credentials
- Encrypted communication with external APIs
- Local file system protection for document storage

**Access Control**:
- API key-based authentication for external services
- Environment variable management for sensitive configuration
- Restricted access to system configuration files

**Incident Response**:
- Monitoring for unusual query patterns or system behavior
- Automated alerts for API failures or security issues
- Incident logging and response procedures

## Human Oversight

*EU AI Act [Article 11](https://artificialintelligenceact.eu/article/11/);; [Annex IV](https://artificialintelligenceact.eu/annex/4/) paragraph 2(e)  
EU AI Act [Article 14](https://artificialintelligenceact.eu/article/14/)*

**Human-in-the-Loop Mechanisms**: The system incorporates human oversight through the following mechanisms:
- Initial system configuration and deployment requires human setup
- Monitor and review system outputs and performance metrics
- Manual intervention capabilities for system configuration updates

**Override and Intervention Procedures**: 
- System administrators can modify routing parameters and thresholds
- Manual restart and recovery procedures for system failures
- Ability to update ethical guidelines and content policies
- Configuration changes for API integrations and dependencies

**User Instructions and Training**: 
- Clear documentation for system deployment and configuration
- Guidelines for monitoring system performance and ethical compliance
- Training materials for understanding system capabilities and limitations

**Limitations and Constraints of the System**: 
- Cannot process multimedia inputs (images, audio, video)
- Limited to text-based natural language queries
- Dependent on external API availability for full functionality
- RAG system limited to pre-indexed document collection
- Cannot perform real-time calculations or code execution
- Language support limitations based on underlying model capabilities

## Incident Management

### Common Issues

**Ethical Analysis Failures**:
- **Problem**: System fails to properly evaluate query ethics
- **Solution**: Review ethical guidelines, update analysis parameters, restart system

**API Connectivity Issues**:
- **Problem**: External APIs (OpenAI, SerpDev) become unavailable
- **Solution**: Implement retry mechanisms, monitor API status, provide fallback options

**RAG System Failures**:
- **Problem**: Vector database or document retrieval fails
- **Solution**: Automatic routing to web search, database integrity checks, system restart

**Support Contact**: to be filled

### Troubleshooting AI Application Deployment

#### Infrastructure-Level Issues

##### Insufficient Resources
- **Problem**: High memory usage due to document embedding processing, CPU bottlenecks during query processing
- **Mitigation Strategy**: 
  - Monitor resource utilization and implement autoscaling
  - Optimize vector database operations and caching
  - Implement rate limiting for query processing

##### Network Failures
- **Problem**: External API connectivity issues affecting web search and language model access
- **Mitigation Strategy**:
  - Test network connectivity to all external services
  - Implement retry mechanisms with exponential backoff
  - Use redundant network paths and failover mechanisms

##### Deployment Pipeline Failures
- **Problem**: Environment configuration issues, dependency conflicts, API key misconfiguration
- **Mitigation Strategy**:
  - Use containerization for environment consistency
  - Implement comprehensive configuration validation
  - Enable detailed logging for deployment diagnostics

#### Integration Problems

##### API Failures
- **Problem**: OpenAI or SerpDev API rate limits, authentication failures, service outages
- **Mitigation Strategy**:
  - Implement circuit breaker patterns and retry logic
  - Monitor API quotas and usage patterns
  - Maintain backup API keys and service alternatives

##### Data Format Mismatches
- **Problem**: Unexpected response formats from external APIs, document parsing errors
- **Mitigation Strategy**:
  - Implement robust input validation and error handling
  - Use schema validation for API responses
  - Maintain backward compatibility for API changes

#### Data Quality Problems
- **Problem**: Outdated smartphone documentation, corrupted vector indices, poor search results
- **Mitigation Strategy**:
  - Implement automated data quality checks
  - Regular document index updates and validation
  - Monitor search result relevance and accuracy

#### Model-Level Issues

##### Performance or Deployment Issues
- **Problem**: Poor routing decisions, inaccurate ethical analysis, slow response times
- **Mitigation Strategy**:
  - Monitor model performance metrics continuously
  - Implement A/B testing for model updates
  - Regular evaluation of decision accuracy

#### Safety and Security Issues

##### Unauthorized Access
- **Problem**: Exposed API keys, unauthorized system access
- **Mitigation Strategy**: to be filled

##### Data Breaches
- **Problem**: Exposure of query data or system configuration
- **Mitigation Strategy**: to be filled

#### Monitoring and Logging Failures

##### Missing or Incomplete Logs
- **Problem**: Insufficient monitoring data for debugging system issues
- **Mitigation Strategy**: to be filled

#### Recovery and Rollback

##### Rollback Mechanisms
- **Problem**: New deployment introduces system failures or performance degradation
- **Mitigation Strategy**: to be filled

##### Disaster Recovery
- **Problem**: Complete system outage or data corruption
- **Mitigation Strategy**: to be filled

### EU Declaration of Conformity

*EU AI Act [Article 47](https://artificialintelligenceact.eu/article/47/)*

To be filled when applicable and certifications are available.

### Standards Applied

- ISO 31000 Risk Management Framework
- NIST Cybersecurity Framework
- CrewAI Framework Standards and Best Practices
- OpenAI API Usage Guidelines
- EU AI Act Compliance Standards for Limited Risk Systems

## Documentation Metadata

### Template Version
Based on EU AI Act Compliance Documentation Template v1.0

### Documentation Authors

- **to be filled, to be filled:** (Owner / Contributor / Manager)
- **to be filled, to be filled:** (Owner / Contributor / Manager)
- **to be filled, to be filled:** (Owner / Contributor / Manager)
