#!/bin/bash
# Setup BioMoQA RAG as a systemd service that starts on boot

echo "Setting up BioMoQA RAG V2 as a system service..."

# Create logs directory
mkdir -p /home/egaillac/BioMoQA-RAG/logs

# Copy service file to systemd
sudo cp /home/egaillac/BioMoQA-RAG/biomoqa-rag.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable biomoqa-rag

# Start the service now
sudo systemctl start biomoqa-rag

# Wait a moment
sleep 3

# Check status
sudo systemctl status biomoqa-rag

echo ""
echo "✓ Service installed!"
echo ""
echo "Commands:"
echo "  Start:   sudo systemctl start biomoqa-rag"
echo "  Stop:    sudo systemctl stop biomoqa-rag"
echo "  Restart: sudo systemctl restart biomoqa-rag"
echo "  Status:  sudo systemctl status biomoqa-rag"
echo "  Logs:    tail -f logs/api_service.log"
echo ""
echo "Service will auto-start on VM reboot."
echo "API available at: http://egaillac.lan.text-analytics.ch:9000"
