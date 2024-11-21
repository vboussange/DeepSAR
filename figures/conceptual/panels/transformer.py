import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_transformer_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Encoder
    for i in range(6):
        # Encoder blocks
        encoder_block = patches.FancyBboxPatch((0.1, 0.8-i*0.12), 0.3, 0.1,
                                               boxstyle="round,pad=0.05", edgecolor='black', facecolor='lightblue')
        ax.add_patch(encoder_block)
        plt.text(0.25, 0.85-i*0.12, f'Encoder Layer {i+1}', ha='center', va='center')

    # Decoder
    for i in range(6):
        # Decoder blocks
        decoder_block = patches.FancyBboxPatch((0.6, 0.8-i*0.12), 0.3, 0.1,
                                               boxstyle="round,pad=0.05", edgecolor='black', facecolor='lightcoral')
        ax.add_patch(decoder_block)
        plt.text(0.75, 0.85-i*0.12, f'Decoder Layer {i+1}', ha='center', va='center')

    # Multi-Head Attention and Feed-Forward in Encoder
    for i in range(6):
        attention_block = patches.FancyBboxPatch((0.15, 0.8-i*0.12), 0.1, 0.02,
                                                 boxstyle="round,pad=0.05", edgecolor='black', facecolor='orange')
        ax.add_patch(attention_block)
        ff_block = patches.FancyBboxPatch((0.25, 0.8-i*0.12), 0.1, 0.02,
                                          boxstyle="round,pad=0.05", edgecolor='black', facecolor='green')
        ax.add_patch(ff_block)
        plt.text(0.2, 0.815-i*0.12, 'MHA', ha='center', va='center', fontsize=8)
        plt.text(0.3, 0.815-i*0.12, 'FF', ha='center', va='center', fontsize=8)

    # Multi-Head Attention, Masked Multi-Head Attention, and Feed-Forward in Decoder
    for i in range(6):
        masked_attention_block = patches.FancyBboxPatch((0.65, 0.8-i*0.12), 0.1, 0.02,
                                                        boxstyle="round,pad=0.05", edgecolor='black', facecolor='purple')
        ax.add_patch(masked_attention_block)
        attention_block = patches.FancyBboxPatch((0.75, 0.8-i*0.12), 0.1, 0.02,
                                                 boxstyle="round,pad=0.05", edgecolor='black', facecolor='orange')
        ax.add_patch(attention_block)
        ff_block = patches.FancyBboxPatch((0.85, 0.8-i*0.12), 0.1, 0.02,
                                          boxstyle="round,pad=0.05", edgecolor='black', facecolor='green')
        ax.add_patch(ff_block)
        plt.text(0.7, 0.815-i*0.12, 'MMHA', ha='center', va='center', fontsize=8)
        plt.text(0.8, 0.815-i*0.12, 'MHA', ha='center', va='center', fontsize=8)
        plt.text(0.9, 0.815-i*0.12, 'FF', ha='center', va='center', fontsize=8)

    # Input and Output Embeddings
    input_embed = patches.FancyBboxPatch((0.05, 0.9), 0.15, 0.05,
                                         boxstyle="round,pad=0.05", edgecolor='black', facecolor='yellow')
    ax.add_patch(input_embed)
    plt.text(0.125, 0.925, 'Input Embeddings', ha='center', va='center')

    output_embed = patches.FancyBboxPatch((0.8, 0.9), 0.15, 0.05,
                                          boxstyle="round,pad=0.05", edgecolor='black', facecolor='yellow')
    ax.add_patch(output_embed)
    plt.text(0.875, 0.925, 'Output Embeddings', ha='center', va='center')

    # Arrows
    for i in range(6):
        # Encoder arrows
        ax.annotate("", xy=(0.1, 0.8-i*0.12), xytext=(0.1, 0.92-i*0.12),
                    arrowprops=dict(arrowstyle="->"))
        # Decoder arrows
        ax.annotate("", xy=(0.6, 0.8-i*0.12), xytext=(0.6, 0.92-i*0.12),
                    arrowprops=dict(arrowstyle="->"))

    # Encoder-Decoder attention arrows
    for i in range(6):
        ax.annotate("", xy=(0.4, 0.8-i*0.12), xytext=(0.6, 0.8-i*0.12),
                    arrowprops=dict(arrowstyle="->", linestyle='dashed'))

    # Final output arrow
    ax.annotate("", xy=(0.75, 0.2), xytext=(0.875, 0.9),
                arrowprops=dict(arrowstyle="->"))

    plt.axis('off')
    plt.show()

draw_transformer_architecture()
