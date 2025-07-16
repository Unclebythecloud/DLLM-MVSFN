import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityAttentionFusionLayer(nn.Module):
    def __init__(self):
        super(SimilarityAttentionFusionLayer, self).__init__()
        self.fusion = nn.Linear(768 * 3, 768)

    def forward(self, content_cls_token, analyze_cls_token, comments_cls_token):
       
        content_analyze_similarity = F.cosine_similarity(content_cls_token, analyze_cls_token, dim=1).unsqueeze(1)
        content_comments_similarity = F.cosine_similarity(content_cls_token, comments_cls_token, dim=1).unsqueeze(1)
        analyze_comments_similarity = F.cosine_similarity(analyze_cls_token, comments_cls_token, dim=1).unsqueeze(1)

        
        similarity_weights = torch.cat((content_analyze_similarity, content_comments_similarity, analyze_comments_similarity), dim=1)
        similarity_weights = F.softmax(similarity_weights, dim=1)  # Softmax确保权重和为1

       
        weighted_content = similarity_weights[:, 0].unsqueeze(1) * content_cls_token
        weighted_analyze = similarity_weights[:, 1].unsqueeze(1) * analyze_cls_token
        weighted_comments = similarity_weights[:, 2].unsqueeze(1) * comments_cls_token

       
        fused_features = torch.cat((weighted_content, weighted_analyze, weighted_comments), dim=1)  # [batch_size, 768 * 3]
        fused_features = self.fusion(fused_features)  # [batch_size, 768]
        return fused_features
    
class AttentionFusionLayer(nn.Module):
    def __init__(self):
        super(AttentionFusionLayer, self).__init__()
        self.fusion = nn.Linear(768 * 3, 768)

       
        self.attention_weights = nn.Linear(768*3, 3)  # 生成三个权重

    def forward(self, content_cls_token, analyze_cls_token, comments_cls_token):
       
        combined_features = torch.cat((content_cls_token, analyze_cls_token, comments_cls_token), dim=1)  # [batch_size, 768 * 3]

        
        attention_scores = self.attention_weights(combined_features)  # [batch_size, 3]
        attention_weights = F.softmax(attention_scores, dim=1)  # Softmax归一化，确保权重和为1

       
        weighted_content = attention_weights[:, 0].unsqueeze(1) * content_cls_token
        weighted_analyze = attention_weights[:, 1].unsqueeze(1) * analyze_cls_token
        weighted_comments = attention_weights[:, 2].unsqueeze(1) * comments_cls_token

        
        fused_features = self.fusion(torch.cat((weighted_content, weighted_analyze, weighted_comments), dim=1))  # [batch_size, 768]
        return fused_features
    
class GatedFusionLayer(nn.Module):
    def __init__(self):
        super(GatedFusionLayer, self).__init__()
        
        self.gate_content = nn.Linear(768, 1)
        self.gate_analyze = nn.Linear(768, 1)
        self.gate_comments = nn.Linear(768, 1)

        
        self.fusion = nn.Linear(768 * 3, 768)

    def forward(self, content_cls_token, analyze_cls_token, comments_cls_token):
        
        gate_content = torch.sigmoid(self.gate_content(content_cls_token))  # [batch_size, 1]
        gate_analyze = torch.sigmoid(self.gate_analyze(analyze_cls_token))  # [batch_size, 1]
        gate_comments = torch.sigmoid(self.gate_comments(comments_cls_token))  # [batch_size, 1]

       
        gated_content = gate_content * content_cls_token
        gated_analyze = gate_analyze * analyze_cls_token
        gated_comments = gate_comments * comments_cls_token

       
        fused_features = self.fusion(torch.cat((gated_content, gated_analyze, gated_comments), dim=1))  # [batch_size, 768]
        return fused_features


class SLMModel(nn.Module):
    def __init__(self, bert_model, hidden_size=256, num_labels=2):
        super(SLMModel, self).__init__()
        self.bert = bert_model
        
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        
        self.similarity_fusion_layer = SimilarityAttentionFusionLayer()
        self.attention_fusion_layer = AttentionFusionLayer()
        self.gated_fusion_layer = GatedFusionLayer()
        
        
        self.classifier = nn.Sequential(
            nn.Linear(768*3, hidden_size),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, content_input_ids, content_attention_mask, 
                analyze_input_ids, analyze_attention_mask, 
                comments_input_ids, comments_attention_mask):
        
        content_outputs = self.bert(input_ids=content_input_ids, attention_mask=content_attention_mask)
        analyze_outputs = self.bert(input_ids=analyze_input_ids, attention_mask=analyze_attention_mask)
        comments_outputs = self.bert(input_ids=comments_input_ids, attention_mask=comments_attention_mask)

       
        content_last_hidden_state = content_outputs.last_hidden_state
        analyze_last_hidden_state = analyze_outputs.last_hidden_state
        comments_last_hidden_state = comments_outputs.last_hidden_state
        
        
        content_last_hidden_state = content_last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_dim]
        analyze_last_hidden_state = analyze_last_hidden_state[:, 0, :]
        comments_last_hidden_state = comments_last_hidden_state[:, 0, :]

        
        fused_features_sim = self.similarity_fusion_layer(content_last_hidden_state, analyze_last_hidden_state, comments_last_hidden_state)
        fused_features_att = self.attention_fusion_layer(content_last_hidden_state, analyze_last_hidden_state, comments_last_hidden_state)
        fused_features_gated = self.gated_fusion_layer(content_last_hidden_state, analyze_last_hidden_state, comments_last_hidden_state)
        
        fused_features = torch.cat((fused_features_sim, fused_features_att,fused_features_gated), dim=1)
    
        logits = self.classifier(fused_features)

        return logits
