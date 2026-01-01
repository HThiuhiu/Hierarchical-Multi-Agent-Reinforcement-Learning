# Hierarchical-Multi-Agent-Reinforcement-Learning
Reinforcement Learning
my_project/
├── models/
│   ├── __init__.py
│   ├── attention_manager.py    # Code class Attention của bạn (High-level)
│   └── navigation_skill.py     # Code load và chạy skill di chuyển (Low-level)
├── checkpoints/
│   ├── attention_manager.pth   # File weights của model Attention sau khi train xong
│   └── basic_navigation.pth    # File weights của skill đã có sẵn
├── utils/
│   └── environment.py          # Class môi trường giả lập
├── inference.py                # File chính để chạy demo/test
└── README.md                   # Hướng dẫn sử dụng