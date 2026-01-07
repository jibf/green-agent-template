"""
Response quality evaluator using LLM-as-Judge.

This module evaluates the completeness and correctness of agent responses
using GPT-4o as the judge model.
"""
import json
import os
from typing import Optional
from openai import OpenAI


def retry(max_attempts=10, delay=1):
    """Retry decorator for API calls."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            import asyncio
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        if asyncio.iscoroutinefunction(func):
                            await asyncio.sleep(delay)
                        else:
                            import time
                            time.sleep(delay)
            raise last_error
        return wrapper
    return decorator


def decode_json(text: str) -> dict:
    """Extract JSON from text that may contain markdown code blocks."""
    text = text.strip()

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code block
    if "```json" in text.lower():
        start = text.lower().find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            json_text = text[start:end].strip()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass

    # Try to extract from ``` block
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            json_text = text[start:end].strip()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass

    # Last resort: try to find {...} pattern
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        json_text = text[start:end+1]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    return None


class ResponseEvaluator:
    """
    Evaluates response quality using LLM-as-Judge.

    Evaluates two aspects:
    1. Completeness: Does the response address all parts of the query?
    2. Correctness: Is the response consistent with API results?
    """

    # System prompts from ComplexFuncBench
    COMPLETENESS_SYSTEM_PROMPT = """You are a helpful response completeness detect assistant. Your task is to evaluate the response based on whether it fully addresses all aspects of the user's query.
# Your Task
For each user query and corresponding response, you should determine the completeness of the response using the following criteria:
- If the response covers all requested information and addresses all parts of the user's query, it should be considered complete and receive a score of 2.
- If the response addresses some but not all parts of the user's query, it should be considered partial and receive a score of 1.
- If the response does not address any of the requested information in the user's query, it should be considered incomplete and receive a score of 0.

# Output Format
You should output the score for each user query and corresponding response in JSON format with following keys:
- score: the completeness score for the response (0, 1, or 2)
- reason: a string describing the reason for the score

# Example
input:
query: I'm thinking about heading to Paris. Can you tell me which museums are currently trending?
response: The Louvre is the most trending museums in Paris.

output:
```JSON
{"score": 2, "reason": "all requested information is addressed"}
```
"""

    CORRECTNESS_SYSTEM_PROMPT = """You are a helpful response correctness detect assistant. Your task is to evaluate the response based on its accuracy in matching the details provided by API response.
# Your task
Give a dialogue history containing user query, function calls and api responses, you should determine the correctness of the corresponding response using the following criteria:
- If the response is consistent with the information provided in the API response, it should be considered entirely correct and receive a score of 2.
- If the response partially matches the information provided in the API response (with some correct and some incorrect details), it should be considered partially correct and receive a score of 1.
- If the response does not match any of the information provided in the API response, it should be considered incorrect and receive a score of 0.

# Output Format
You should output the score for each dialogue history and corresponding response in JSON format with following keys:
- score: the correctness score for the response (0, 1, or 2)
- reason: a string describing the reason for the score

## example output
```JSON
{"score": 2, "reason": "All mentioned information is correct."}
```
"""

    def __init__(self, eval_model: str = "gpt-4o-2024-08-06", logger=None):
        """
        Initialize evaluator.

        Args:
            eval_model: Model to use for evaluation (default: gpt-4o-2024-08-06)
            logger: Optional logger instance
        """
        self.eval_model = eval_model
        self.logger = logger

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)

    @retry(max_attempts=10)
    def _call_judge(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """Call the judge model."""
        try:
            completion = self.client.chat.completions.create(
                model=self.eval_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
            )

            response = completion.choices[0].message.content
            result = decode_json(response)

            if self.logger:
                self.logger.info(f"Judge response: {response}")
                self.logger.info(f"Decoded result: {result}")

            return result

        except Exception as e:
            if self.logger:
                self.logger.error(f"Judge call failed: {e}")
            return None

    async def evaluate_completeness(self, query: str, response: str) -> dict:
        """
        Evaluate response completeness.

        Args:
            query: User query
            response: Agent response

        Returns:
            dict with:
                - score: 0/1/2
                - reason: explanation
        """
        if not response or response == "":
            return {
                "score": -2,
                "reason": "No response generated"
            }

        user_prompt = f"""input:
query: {query}
response: {response}

output:
"""

        result = self._call_judge(self.COMPLETENESS_SYSTEM_PROMPT, user_prompt)

        if result and isinstance(result, dict) and "score" in result:
            if result['score'] in [0, 1, 2]:
                return {
                    "score": result['score'],
                    "reason": result.get("reason", "")
                }

        return {
            "score": -1,
            "reason": "Completeness evaluation failed"
        }

    async def evaluate_correctness(self, history: list, response: str) -> dict:
        """
        Evaluate response correctness against API results.

        Args:
            history: Conversation history (list of turns)
            response: Final agent response

        Returns:
            dict with:
                - score: 0/1/2
                - reason: explanation
        """
        if not response or response == "":
            return {
                "score": -2,
                "reason": "No response generated"
            }

        history_str = json.dumps(history, ensure_ascii=False, indent=2)

        user_prompt = f"""dialogue history: {history_str}
response: {response}

output:
"""

        result = self._call_judge(self.CORRECTNESS_SYSTEM_PROMPT, user_prompt)

        if result and isinstance(result, dict) and "score" in result:
            if result['score'] in [0, 1, 2]:
                return {
                    "score": result['score'],
                    "reason": result.get("reason", "")
                }

        # Evaluation failed
        return {
            "score": -1,
            "reason": "Correctness evaluation failed"
        }

    async def evaluate_response(
        self,
        task_data: dict,
        generated_convs: list,
        final_response: Optional[str]
    ) -> dict:
        """
        Evaluate both completeness and correctness.

        Args:
            task_data: Original task data with query
            generated_convs: Generated conversation history
            final_response: Final agent response

        Returns:
            dict with:
                - complete: {score, reason}
                - correct: {score, reason}
        """
        if not final_response:
            return {
                "complete": {"score": -2, "reason": "No response generated"},
                "correct": {"score": -2, "reason": "No response generated"}
            }

        # Extract query from task data
        query = task_data['conversations'][0]['content']

        # Evaluate completeness
        complete_result = await self.evaluate_completeness(query, final_response)

        # Evaluate correctness (use history excluding final response)
        history = [turn for turn in generated_convs if turn.get('role') != 'assistant' or 'content' not in turn]
        correct_result = await self.evaluate_correctness(history, final_response)

        return {
            "complete": complete_result,
            "correct": correct_result
        }
