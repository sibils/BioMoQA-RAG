#!/bin/bash
# Setup V3.1 Fast API as systemd service

set -e

echo "Setting up BioMoQA RAG V3.1 Fast API as systemd service..."
echo

# Create logs directory if it doesn't exist
mkdir -p /home/egaillac/BioMoQA-RAG/logs

# Copy service file
echo "1. Installing service file..."
sudo cp biomoqa-rag-v3-fast.service /etc/systemd/system/

# Reload systemd
echo "2. Reloading systemd..."
sudo systemctl daemon-reload

# Enable service (auto-start on boot)
echo "3. Enabling service (will auto-start on boot)..."
sudo systemctl enable biomoqa-rag-v3-fast

# Start service
echo "4. Starting service..."
sudo systemctl start biomoqa-rag-v3-fast

# Wait a moment
sleep 5

# Check status
echo
echo "5. Service status:"
sudo systemctl status biomoqa-rag-v3-fast --no-pager

echo
echo "========================================"
echo "✓ V3.1 Fast API Service Setup Complete!"
echo "========================================"
echo
echo "Service is now running on 0.0.0.0:9000"
echo
echo "Commands:"
echo "  Start:   sudo systemctl start biomoqa-rag-v3-fast"
echo "  Stop:    sudo systemctl stop biomoqa-rag-v3-fast"
echo "  Restart: sudo systemctl restart biomoqa-rag-v3-fast"
echo "  Status:  sudo systemctl status biomoqa-rag-v3-fast"
echo "  Logs:    sudo journalctl -u biomoqa-rag-v3-fast -f"
echo
echo "Access:"
echo "  Internal: http://localhost:9000"
echo "  External: http://<your-vm-ip>:9000"
echo
echo "Test:"
echo "  curl http://localhost:9000/health"
echo
