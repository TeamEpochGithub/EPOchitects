import matplotlib.pyplot as plt
from pathlib import Path

def plot_prediction(result, task, task_id):
    # Get training examples for this task
    train_examples = task['train']
    num_train = len(train_examples)
    
    # Calculate figure size and layout
    # Height is proportional to number of examples plus one row for test/prediction
    fig_height = 5 * (num_train + 1)
    plt.figure(figsize=(15, fig_height))
    
    # Plot training examples first
    for i, example in enumerate(train_examples):
        # Input
        plt.subplot(num_train + 1, 3, i * 3 + 1)
        plt.imshow(example['input'])
        plt.title(f'Training Input {i+1}')
        plt.axis('off')
        
        # Expected Output
        plt.subplot(num_train + 1, 3, i * 3 + 3)
        plt.imshow(example['output'])
        plt.title(f'Training Output {i+1}')
        plt.axis('off')
    
    # Plot test/prediction on the last row
    # Test Input
    plt.subplot(num_train + 1, 3, num_train * 3 + 1)
    plt.imshow(task['test'][0]['input'])
    plt.title('Test Input')
    plt.axis('off')
    
    # Prediction
    plt.subplot(num_train + 1, 3, num_train * 3 + 2)
    plt.imshow(result)
    plt.title('Prediction')
    plt.axis('off')
    
    # Expected Output
    plt.subplot(num_train + 1, 3, num_train * 3 + 3)
    plt.imshow(task['test'][0]['output'])
    plt.title('Expected Output')
    plt.axis('off')
    
    plt.suptitle(f'Task {task_id}')
    plt.tight_layout()
    
    # Create visuals directory in output folder
    visuals_dir = Path(__file__).resolve().parent.parent.parent / 'output' 
    
    # Save figure
    fig_path = visuals_dir / f"{task_id.replace('.json', '.png')}"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory
        
    return
