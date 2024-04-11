import os
from typing import Any, AsyncGenerator, Optional, Union

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI

from approaches.approach import Approach, ThoughtStep
from core.authentication import AuthenticationHelper
from core.messagebuilder import MessageBuilder

# Replace these with your own values, either in environment variables or directly here
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")


class RetrieveThenReadApproach(Approach):
    """
    Simple retrieve-then-read implementation, using the AI Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    system_chat_template = ("""you are A SQL developer	
USING this SCHEMA FOR tables IN an azure SQL DATABASE	
CREATE TABLE [rnq_ai].[vehicle_repair_copilot](	
	[unique_key] [NVARCHAR](MAX) NULL,
	[claim_causal_part_25_code] [NVARCHAR](MAX) NULL,
	[claim_causal_part_description] [NVARCHAR](MAX) NULL,
	[claim_claim_id] [NVARCHAR](MAX) NULL,
	[claim_claim_item_id] [NVARCHAR](MAX) NULL,
	[claim_claim_number] [NVARCHAR](MAX) NULL,
	[claim_claim_received_date] [DATE] NULL,
	[claim_condition_code] [NVARCHAR](MAX) NULL,
	[claim_condition_description] [NVARCHAR](MAX) NULL,
	[claim_days_down] [FLOAT] NULL,
	[claim_fail_date] [DATE] NULL,
	[claim_group_noun_id] [NVARCHAR](MAX) NULL,
	[claim_labor_spend] [FLOAT] NULL,
	[claim_mileage] [INT] NULL,
	[claim_net_spend] [FLOAT] NULL,
	[claim_noun_code] [NVARCHAR](MAX) NULL,
	[claim_noun_description] [NVARCHAR](MAX) NULL,
	[claim_noun_group_code] [NVARCHAR](MAX) NULL,
	[claim_noun_group_description] [NVARCHAR](MAX) NULL,
	[claim_orgainziation_noun] [NVARCHAR](MAX) NULL,
	[claim_parts_net_total] [FLOAT] NULL,
	[claim_process_date] [DATE] NULL,
	[claim_supplier_cd] [NVARCHAR](MAX) NULL,
	[claim_supplier_code_description] [NVARCHAR](MAX) NULL,
	[claim_technician_comment] [NVARCHAR](MAX) NULL,
	[claim_warranty_policy_code] [NVARCHAR](MAX) NULL,
	[claim_warranty_policy_description] [NVARCHAR](MAX) NULL,
	[dms_cause_description] [NVARCHAR](MAX) NULL,
	[dms_complaint_description] [NVARCHAR](MAX) NULL,
	[dms_correction_description] [NVARCHAR](MAX) NULL,
	[dms_job_number] [NVARCHAR](MAX) NULL,
	[dms_job_start_date] [DATETIME] NULL,
	[dms_job_end_date] [DATETIME] NULL,
	[dms_repair_order_id] [VARCHAR](20) NULL,
	[dms_technician_notes] [NVARCHAR](MAX) NULL,
	[dms_warranty_indicator] [NVARCHAR](MAX) NULL,
	[sales_country] [NVARCHAR](MAX) NULL,
	[sales_dealer_city] [NVARCHAR](MAX) NULL,
	[sales_dealer_code] [NVARCHAR](MAX) NULL,
	[sales_dealer_name] [NVARCHAR](MAX) NULL,
	[sales_dealer_state] [NVARCHAR](MAX) NULL,
	[sales_engine_build_date] [DATE] NULL,
	[sales_engine_family] [NVARCHAR](MAX) NULL,
	[sales_engine_family_description] [NVARCHAR](MAX) NULL,
	[sales_engine_model_year] [INT] NULL,
	[sales_engine_plant_code] [NVARCHAR](MAX) NULL,
	[sales_engine_plant_description] [NVARCHAR](MAX) NULL,
	[sales_engine_serial] [NVARCHAR](MAX) NULL,
	[sales_engine_type_description] [NVARCHAR](MAX) NULL,
	[sales_fleet_customer_number] [NVARCHAR](MAX) NULL,
	[sales_fleet_name] [NVARCHAR](MAX) NULL,
	[sales_months_in_service] [INT] NULL,
	[sales_months_in_service_include_code] [NVARCHAR](MAX) NULL,
	[sales_special_warranty_code] [NVARCHAR](MAX) NULL,
	[sales_vehicle_application_family_code] [NVARCHAR](MAX) NULL,
	[sales_vehicle_application_type] [NVARCHAR](MAX) NULL,
	[sales_vehicle_model_family_code] [NVARCHAR](MAX) NULL,
	[sales_vehicle_model_year] [INT] NULL,
	[sales_vehicle_plant_code] [NVARCHAR](MAX) NULL,
	[sales_vehicle_plant_description] [NVARCHAR](MAX) NULL,
	[sales_vehicle_prod_date] [DATE] NULL,
	[sales_vehicle_series_code] [NVARCHAR](MAX) NULL,
	[sales_vehicle_series_description] [NVARCHAR](MAX) NULL,
	[sales_vin] [VARCHAR](17) NULL,
	[sales_vin_model_code] [NVARCHAR](MAX) NULL,
	[sales_vin8] [NVARCHAR](MAX) NULL,
	[sales_warranty_start_date] [DATE] NULL,
	[high_watermark_date] [DATETIME] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]	
GO	
	
The fields claim_causal_part_description ,claim_causal_part_25_code and claim_noun_description may be referred to as part or noun	
	
The fields sales_engine_family AND sales_engine_family_description may be referred TO AS engine family, engine family NAME, eng family NAME or eng family	
	
The field sales_vin8 may be referred TO AS Vehicle Identification Number, VIN, vin, vin8,Vehicle,  chassis or chassis number.  IN ALL cases USE the LAST 8 characters FOR sales_vin8	
	
The field sales_vehicle_series_code may contain Model series, series or series TYPE	
	
The field sales_engine_type_description may be referred TO AS engine type		
	
queries use sales_vehicle_prod_date as the date field unless calculating spend then use claim_process_date	
	
TO WRITE SQL queries.	

Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, for example [info1.txt]. Don't combine sources, list each source separately, for example [info1.txt][info2.pdf]
"""
    )

    # shots/sample conversation
    question = """
'What is the deductible for the employee plan for a visit to Overlake in Bellevue?'

Sources:
info1.txt: deductibles depend on whether you are in-network or out-of-network. In-network deductibles are $500 for employee and $1000 for family. Out-of-network deductibles are $1000 for employee and $2000 for family.
info2.pdf: Overlake is in-network for the employee plan.
info3.pdf: Overlake is the name of the area that includes a park and ride near Bellevue.
info4.pdf: In-network institutions include Overlake, Swedish and others in the region
"""
    answer = "In-network deductibles are $500 for employee and $1000 for family and Overlake is in-network for the employee plan."

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_model: str,
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.embedding_model = embedding_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller

    async def run(
        self,
        messages: list[dict],
        stream: bool = False,  # Stream is not used in this approach
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> Union[dict[str, Any], AsyncGenerator[dict[str, Any], None]]:
        q = messages[-1]["content"]
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = overrides.get("semantic_ranker") and has_text

        use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        top = overrides.get("top", 3)
        filter = self.build_filter(overrides, auth_claims)
        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if has_vector:
            vectors.append(await self.compute_text_embedding(q))

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        query_text = q if has_text else None

        results = await self.search(top, query_text, filter, vectors, use_semantic_ranker, use_semantic_captions)

        user_content = [q]

        template = overrides.get("prompt_template", self.system_chat_template)
        model = self.chatgpt_model
        message_builder = MessageBuilder(template, model)

        # Process results
        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)

        # Append user message
        content = "\n".join(sources_content)
        user_content = q + "\n" + f"Sources:\n {content}"
        message_builder.insert_message("user", user_content)
        #message_builder.insert_message("assistant", self.answer)
        #message_builder.insert_message("user", self.question)
        updated_messages = message_builder.messages
        chat_completion = (
            await self.openai_client.chat.completions.create(
                # Azure Open AI takes the deployment name as the model name
                model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
                messages=updated_messages,
                temperature=overrides.get("temperature", 0.3),
                max_tokens=1024,
                n=1,
            )
        ).model_dump()

        data_points = {"text": sources_content}
        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Search using user query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "has_vector": has_vector,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    [str(message) for message in updated_messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
            ],
        }

        chat_completion["choices"][0]["context"] = extra_info
        chat_completion["choices"][0]["session_state"] = session_state
        return chat_completion
