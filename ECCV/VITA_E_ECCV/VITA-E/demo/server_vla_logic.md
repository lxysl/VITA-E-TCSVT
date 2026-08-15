# server.py
┌前端---------------------------------┐
|   send_audio    |    send_video    |
└------------------------------------┘


┌后端---------------------------------┐
|   video is a deque with maxlen=8   |
|                                    |
|      handle_audio                  |
|            ↓                       |
|      pcm_fifo_queue                |
|            ↓ (not empty)           |
|      wakeup_and_vad                |
|            ↓ (dialog+video)        |
|      request_input_queue           |
|            ↓ (not empty)           |
|      llm.generate                  |
|            ↓                       |
|      tts.decode                    |
|            ↓                       |
|      socketio.emit(audio)          |
|                                    |
└------------------------------------┘

`start_event` is a semaphore, `start_event` in model A is `other_start_event` in model B
`stop_event` is a semaphore, `stop_event` in model A is `other_stop_event` in model B

```python
while True:
    if not request_input_queue.empty():
        if start_event == True:
            input = request_input_queue.get()
            other_start_event = True
            start_event = False
        output = llm.generate(input)
        for token in output:
            if is_first_time_to_work == True:
                stop_event = False
                other_stop_event = True
                output_queue.clear()
                is_first_time_to_work = False
            if stop_event == False:
                output_queue.put(token)
```

---

# server_vla.py

add `current_dialog` and `current_video_frame` to the global params
add `is_generating_action_feature` to model params as a semaphore, `is_generating_action_feature` in model A is `other_is_generating_action_feature` in model B

* 模型输出的特殊Token：

`[RESPONSE]` 首token，表示输出语音/文字，语音用☞表示，文字用☜表示，如果是语音，则需要过TTS解码，如果是文字，则直接输出

`[ACT]` 首token，表示输出动作特征，用☝表示

`[HALT]` 首token，表示急停，用☀表示

`[INSTRUCTION]` 中间token，表示前面是语音或文字回复，后面接输出的动作指令，用☯表示，其对应的首token一定是[ACT]或[HALT]

`[END]` 首token，表示动作结束，语音用☞表示，文字用☜表示，如果是语音，则需要过TTS解码，如果是文字，则直接输出

| 功能             | 输入                                              | 语音输出                                                                 | 动作输出     |
|------------------|--------------------------------------------------|--------------------------------------------------------------------------|--------------|
| 语音回复         | What's on the table                               | [RESPONSE] There are xxx on the table                                    | 不执行       |
| 正常执行         | Pick up and place the toy in the box.             | [ACT] I will pick up the toy and place in the box. [INSTRUCTION] Pick up and place the toy in the box. | 用[INSTRUCTION]后面的 prompt 执行动作     |
| 物品不存在不执行 | Grasp the red bottle                              | [RESPONSE] Object does not exist                                         | 不执行       |
| 动作打断         | Stop!                                             | [HALT] Stop the action.                             
| 动作打断与回撤   | (Action 1 is running) Stop! Do action 2.          | [ACT] OK, I will do action 2. [INSTRUCTION] Do action 2.                       | 停止当前动作（打断另一个模型的动作特征生成） |
| 边对话边执行     | Pick up and place the toy in the box.             | … a black bottle on the table…                                           | 抓取         |
| 语音打断         | What’s on the table.                              | A→B: [RESPONSE] There is a green toy on the table                        | 不执行       |

```python
global current_dialog, current_video_frame, action_feature_stack  # 全局的共享变量
last_dialog, last_video_frame = None, None  # 记录上一次的对话和视频帧
while True:
    time.sleep(0.01)
    if not request_input_queue.empty() and is_generating_action_feature == False:
        # 有输入时，且没有在生成动作特征
        if start_event == True:
            # start_event为True，表示允许生成
            inputs = request_input_queue.get()
            if other_is_generating_action_feature == False:
                # 另一个模型没有在生成动作特征，则下次让对方生成，自己不生成
                other_start_event = True
                start_event = False
        else:
            # 自己不被允许生成
            continue
        outputs = llm.generate(inputs)  # 生成输出
        # 遍历生成的token
        is_first_time_to_work = True
        for token in outputs:
            if is_first_time_to_work == True:  # 检查第一个token的情况
                stop_event = False  # 允许把自己的输出发给下游，不被打断
                if other_is_generating_action_feature == False:
                    other_stop_event = True  # 如果对方没有在生成动作特征，可能在输出语音，则打断对方（语音打断）
                if token == [HALT]:
                    other_stop_event = True  # 如果首 token 是急停，则打断对方（急停打断），对方在[TAG1]处被打断
                if token == [ACT]:
                    is_generating_action_feature = True  # 如果首 token 是要输出动作，则开始生成动作特征
                    if other_is_generating_action_feature == False:
                        action_feature_stack.clear() # 如果对方没在生成，这是个新动作，清空栈
                    try:
                        current_dialog = re.search(r"[INSTRUCTION](.*)$", outputs).group(1).strip()  # 获取动作指令，并赋值给共享变量
                    except:
                        warnings.warn("No [INSTRUCTION] found in outputs when [ACT] is found")
                        current_dialog = inputs
                output_queue.clear()  # 清空输出队列，准备输出自己的 token
                is_first_time_to_work = False
            if stop_event == False:
                if token == [INSTRUCTION]:
                    break  # 不向下游语音队列输出[INSTRUCTION]后面的 token
                output_queue.put(token)  # 向下游语音队列输出自己的 token
    elif is_generating_action_feature == True:
        if other_is_generating_action_feature == True:
            other_stop_event = True  # 如果对方在生成动作特征，则打断对方（动作打断），对方在[TAG1]处被打断
            other_is_generating_action_feature = False
            other_start_event = True  # 打断对方以后，下次让对方生成，自己不生成，自己进入动作特征生成阶段
            start_event = False  # 自己下次不被允许生成
        if stop_event == True:  # [TAG1] 检查是否被打断
            is_generating_action_feature = False  # 如果被打断，则停止生成动作特征
            stop_event = False  # 如果被打断，则下次允许生成自己的 token
        if llm.decode(outputs)[-1] == [END]:
            is_generating_action_feature = False  # 如果动作结束，则停止生成动作特征
            stop_event = False  # 如果动作结束，则下次允许生成自己的 token

        if is_generating_action_feature == True:  # 如果还在生成动作特征，没有被打断
            if len(action_feature_stack) > 0:  # 如果栈不为空，则执行回撤
                retraction_action = action_feature_stack.pop()
                socketio.emit('action_feature', retraction_action)
                continue
            if last_dialog != current_dialog or last_video_frame != current_video_frame:
                inputs = (current_dialog, current_video_frame)  # 拿到当前的指令和视频帧
                last_dialog, last_video_frame = current_dialog, current_video_frame
            else:
                inputs = None
            outputs = llm.forward(inputs)  # 生成动作特征
            action_feature_stack.append(outputs) # 将动作特征压入栈
            socketio.emit('action_feature', outputs)  # 输出动作特征
        else:  # 如果被打断，则输出急停
            socketio.emit('action_feature', HALT)  # 输出急停
```

