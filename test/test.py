import torch

print("PyTorch version:", torch.__version__)
print("MPS (Metal Performance Shaders) available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

if torch.backends.mps.is_available():
    print("✅ MPS is available and ready to use!")
    
    # Test MPS functionality
    device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Create a tensor on MPS
    x = torch.rand(5, 3, device=device)
    print("Tensor created on MPS:")
    print(x)
    
    # Test basic operations
    y = x + 1
    print("Tensor after adding 1:")
    print(y)
    
    # Test matrix multiplication
    z = torch.mm(x, x.t())
    print("Matrix multiplication result:")
    print(z)
    
    print("✅ MPS is working correctly!")
else:
    print("❌ MPS is not available")
    print("This could be because:")
    print("- You're not on macOS 12.3+")
    print("- You don't have an Apple Silicon Mac (M1/M2/M3)")
    print("- PyTorch was not compiled with MPS support")
    
    # Fallback to CPU
    print("\nFalling back to CPU:")
    x = torch.rand(5, 3)
    print(x)
