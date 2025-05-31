import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer

class CLIPPromptClassifier(nn.Module):
    def __init__(self,
                 clip_model_name="openai/clip-vit-base-patch16",
                 n_ctx=5,
                 classnames=["a photo of a real face", "a photo of a fake face"]):
        super().__init__()

        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        self.n_ctx = n_ctx
        self.ctx_dim = self.text_encoder.config.hidden_size
        self.classnames = classnames
        self.n_classes = len(classnames)

        # Learnable prompt context
        self.ctx = nn.Parameter(torch.randn(self.n_ctx, self.ctx_dim))

        # Tokenized class name prompts
        tokenized = self.tokenizer(classnames, return_tensors="pt", padding=True, truncation=True)
        self.register_buffer("prompt_ids", tokenized.input_ids)
        self.register_buffer("prompt_attention_mask", tokenized.attention_mask)

    def encode_text_with_prompt(self):
        device = self.ctx.device
        input_ids = self.prompt_ids.to(device)
        embed = self.text_encoder.get_input_embeddings()(input_ids)  # [n_class, seq_len, 512]

        # Remove BOS and EOS
        embed_cls = self.text_encoder.get_input_embeddings().weight[self.tokenizer.bos_token_id].unsqueeze(0).unsqueeze(0).expand(self.n_classes, 1, -1)
        embed_eos = self.text_encoder.get_input_embeddings().weight[self.tokenizer.eos_token_id].unsqueeze(0).unsqueeze(0).expand(self.n_classes, 1, -1)
        embed_class_tokens = embed[:, 1:-1, :]  # remove BOS and EOS

        # Final prompt = [CLS] + ctx + class name tokens + [EOS]
        full_prompt_embed = torch.cat([
            embed_cls,
            self.ctx.unsqueeze(0).expand(self.n_classes, -1, -1),
            embed_class_tokens,
            embed_eos
        ], dim=1)  # [n_class, seq_len, 512]

        attention_mask = torch.ones(full_prompt_embed.shape[:2], dtype=torch.long, device=device)
        output = self.text_encoder(inputs_embeds=full_prompt_embed, attention_mask=attention_mask)
        text_features = output.last_hidden_state[:, 0, :]  # <CLS>
        return F.normalize(text_features, dim=-1)

    def encode_image(self, pixel_values):
        image_output = self.vision_encoder(pixel_values=pixel_values)
        image_features = image_output.last_hidden_state[:, 0, :]  # CLS token
        return F.normalize(image_features, dim=-1)

    def forward(self, pixel_values):
        image_feat = self.encode_image(pixel_values)
        text_feat = self.encode_text_with_prompt()
        logits = image_feat @ text_feat.T  # [B, 2]
        return logits