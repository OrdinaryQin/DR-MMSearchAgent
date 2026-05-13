# Modified version of WebImageToImageSearchTool
# Title priority: if title's index corresponds to HTML or missing image, return text only (no placeholder image)

import os
import json
from typing import Any, Optional, Tuple
from PIL import Image
import numpy as np
from uuid import uuid4
import re
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse


class WebImageToImageSearchTool(BaseTool):
    """Web image to image search tool with title-priority logic.

    If the title at index i corresponds to an HTML entry or the image does not exist,
    return text only without providing placeholder images.
    """

    def __init__(self, config: dict = None, tool_schema: OpenAIFunctionToolSchema = None):
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema(
                type="function",
                function={
                    "name": "web_image_to_image_search",
                    "description": "Searches for relevant images based on the original image using web search.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            )
        super().__init__(config or {}, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)
        cache_id = create_kwargs.get("ids")
        if cache_id is None:
            raise ValueError("Missing required 'ids' parameter in create_kwargs")
        if not hasattr(self, '_instance_dict'):
            self._instance_dict = {}
        self._instance_dict[instance_id] = {
            "cache_id": cache_id,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, dict]:
        instance_data = {}
        if hasattr(self, '_instance_dict') and instance_id in self._instance_dict:
            instance_data = self._instance_dict[instance_id]
        cache_id = instance_data.get("cache_id")
        try:
            tool_returned_str, tool_returned_images, tool_stat = self.call_image_search(cache_id)
            response = ToolResponse(
                text=tool_returned_str,
                image=tool_returned_images if tool_returned_images else None
            )
            reward = 0.0
            return response, reward, tool_stat
        except Exception as e:
            error_msg = f"[Image Search Results] Error executing image search: {str(e)}"
            return (
                ToolResponse(text=error_msg),
                -0.1,
                {"success": False, "error": str(e)}
            )

    def _open_image_safe(self, img_path: str) -> Image.Image:
        try:
            img = Image.open(img_path)
            img.load()
            return img
        except Exception:
            return None

    def _normalize_cache_id(self, cache_id: str) -> str:
        if cache_id.startswith("fvqa_train_") or cache_id.startswith("fvqa_test_"):
            return cache_id
        m = re.match(r'^(\d+_\d+)', cache_id)
        return m.group(1) if m else cache_id

    def call_image_search(self, cache_id: str, max_images: int = -1):
        tool_returned_images = []
        tool_returned_str = ""
        tool_success = False
        cache_path = None
        test_cache_base = None

        target_size = (448, 448)

        try:
            cache_id = self._normalize_cache_id(cache_id)
            fvqa_train_cache_base = self.config.get("fvqa_train_cache_path", "fvqa_train_cache")
            all_train_cache_base = self.config.get("all_train_cache_path", "fvqa_train_cache")
            test_cache_base = self.config.get("test_cache_path", "fvqa_test_cache")
            default_max_images = self.config.get("max_images", 3)

            if max_images <= 0:
                max_images = default_max_images

            if cache_id.startswith("fvqa_train_"):
                cache_path = os.path.join(fvqa_train_cache_base, cache_id)
            elif cache_id.startswith("fvqa_test_"):
                cache_path = os.path.join(test_cache_base, cache_id)
            else:
                cache_path = os.path.join(all_train_cache_base, cache_id)

            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Cache folder {cache_path} does not exist")

            meta_file = os.path.join(cache_path, "meta.json")
            if not os.path.exists(meta_file):
                raise FileNotFoundError(f"Meta file {meta_file} does not exist")

            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)

            if "search_results" in meta_data:
                title_list = [item.get("title", "") for item in meta_data.get("search_results", [])]
                image_urls = [item.get("image_url", "") for item in meta_data.get("search_results", [])]
            else:
                title_list = meta_data.get("title_list", [])
                image_urls = meta_data.get("image_urls", [])

            tool_returned_str = "[Image Search Results]: \n"
            collected_images_count = 0

            for i, (title, _) in enumerate(zip(title_list, image_urls)):
                html_filename = f"img_{i:03d}.html"
                html_path = os.path.join(cache_path, html_filename)

                # --- Title priority: if HTML exists, return text only ---
                if os.path.exists(html_path):
                    tool_returned_str += f"{collected_images_count + 1}. title: {title}\n"
                    collected_images_count += 1
                    if collected_images_count >= max_images:
                        break
                    continue

                # Try to read image
                img_found = False
                img_obj = None
                for ext in [".jpg", ".png", ".webp"]:
                    img_filename = f"img_{i:03d}{ext}"
                    img_path = os.path.join(cache_path, img_filename)
                    if os.path.exists(img_path):
                        img_obj = self._open_image_safe(img_path)
                        if img_obj is not None:
                            img_obj = img_obj.resize(target_size, Image.LANCZOS)
                            img_found = True
                        break

                # If no image found → title only (NO placeholder)
                if not img_found:
                    tool_returned_str += f"{collected_images_count + 1}. title: {title}\n"
                else:
                    tool_returned_images.append(img_obj)
                    tool_returned_str += f"{collected_images_count + 1}. title: {title}\n<image>\n"

                collected_images_count += 1
                if collected_images_count >= max_images:
                    break

            tool_success = True

        except Exception as e:
            tool_returned_str = "[Image Search Results] Error encountered."
            tool_returned_images = []
            tool_success = False

        tool_stat = {
            "success": tool_success,
            "num_images": len(tool_returned_images),
            "requested_max_images": max_images,
            "cache_id": cache_id,
            "cache_path": cache_path,
            "test_cache_base": test_cache_base,
        }

        return tool_returned_str, tool_returned_images, tool_stat

    async def release(self, instance_id: str, **kwargs) -> None:
        if hasattr(self, '_instance_dict') and instance_id in self._instance_dict:
            del self._instance_dict[instance_id]