#include "SttClient.h"

#include <WiFi.h>
#include <esp_heap_caps.h>

#include "config/WifiConfig.h"
#include "drivers/Speaker.h"

namespace
{
constexpr char MULTIPART_BOUNDARY[] = "----RockyBoundary";
constexpr char MULTIPART_PREFIX[] =
    "------RockyBoundary\r\n"
    "Content-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\n"
    "Content-Type: audio/wav\r\n\r\n";
constexpr char MULTIPART_SUFFIX[] = "\r\n------RockyBoundary--\r\n";
constexpr size_t WAV_HEADER_SIZE = 44;
constexpr size_t PCM_WRITE_CHUNK_BYTES = 1024;
constexpr size_t MAX_RESPONSE_BYTES = 1024;
constexpr uint16_t PCM_CHANNELS = 1;
constexpr uint16_t PCM_BITS_PER_SAMPLE = 16;
constexpr uint32_t PCM_SAMPLE_RATE = 16000;
constexpr char DEVICE_ID[] = "rocky-01";
constexpr char RESPONSE_AUDIO_PATH[] = "/audio/rocky_response.wav";
constexpr size_t RESPONSE_AUDIO_BLOCK_BYTES = 4096;
constexpr size_t MAX_RESPONSE_AUDIO_BYTES = 1024 * 1024;
constexpr uint32_t RESPONSE_AUDIO_SAMPLE_RATE = 32000;

uint16_t readLittleEndian16(const uint8_t* source)
{
    return static_cast<uint16_t>(source[0]) |
           (static_cast<uint16_t>(source[1]) << 8);
}

uint32_t readLittleEndian32(const uint8_t* source)
{
    return static_cast<uint32_t>(source[0]) |
           (static_cast<uint32_t>(source[1]) << 8) |
           (static_cast<uint32_t>(source[2]) << 16) |
           (static_cast<uint32_t>(source[3]) << 24);
}

void writeLittleEndian16(uint8_t* destination, uint16_t value)
{
    destination[0] = static_cast<uint8_t>(value & 0xFF);
    destination[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
}

void writeLittleEndian32(uint8_t* destination, uint32_t value)
{
    destination[0] = static_cast<uint8_t>(value & 0xFF);
    destination[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
    destination[2] = static_cast<uint8_t>((value >> 16) & 0xFF);
    destination[3] = static_cast<uint8_t>((value >> 24) & 0xFF);
}

void appendMultipartField(String& prefix, const char* name, const String& value)
{
    prefix += "------RockyBoundary\r\nContent-Disposition: form-data; name=\"";
    prefix += name;
    prefix += "\"\r\n\r\n";
    prefix += value;
    prefix += "\r\n";
}
} // namespace

bool SttClient::checkHealth()
{
    WiFiClient client;
    client.setTimeout(HTTP_TIMEOUT_MS);

    Serial.printf("[HTTP] server=http://%s:%u\n", STT_SERVER_HOST, STT_SERVER_PORT);
    Serial.println("[HTTP] attempting /health");
    const uint32_t connectionStartedAt = millis();
    if (!client.connect(STT_SERVER_HOST, STT_SERVER_PORT))
    {
        if (millis() - connectionStartedAt >= HTTP_TIMEOUT_MS)
            Serial.println("[HTTP] timeout");
        else
            Serial.println("[HTTP] begin failed");
        Serial.println("[HTTP] status=0");
        Serial.println("[HTTP] body=");
        client.stop();
        return false;
    }

    if (!sendGetRequest(client))
    {
        Serial.println("[HTTP] begin failed");
        Serial.println("[HTTP] status=0");
        Serial.println("[HTTP] body=");
        client.stop();
        return false;
    }

    const int statusCode = readStatusCode(client);
    const String body = readResponseBody(client);
    client.stop();

    if (statusCode == 0)
        Serial.println("[HTTP] timeout");
    Serial.printf("[HTTP] status=%d\n", statusCode);
    Serial.printf("[HTTP] body=%s\n", body.c_str());
    return statusCode == 200 && isHealthyResponse(body);
}

bool SttClient::transcribe(const int16_t* pcm16, size_t pcmByteCount, BackendResponse& response)
{
    WiFiClient client;
    client.setTimeout(HTTP_TIMEOUT_MS);

    Serial.println("[STT] sending speech...");
    if (!client.connect(STT_SERVER_HOST, STT_SERVER_PORT) ||
        !sendMultipartRequest(client, pcm16, pcmByteCount))
    {
        Serial.println("[STT] error=connection");
        client.stop();
        return false;
    }

    const int statusCode = readStatusCode(client);
    const String body = readResponseBody(client);
    client.stop();

    if (statusCode != 200)
    {
        Serial.printf("[STT] HTTP error=%d\n", statusCode);
        return false;
    }

    if (!extractBackendResponse(body, response))
    {
        return false;
    }

    Serial.println("[STT] HTTP 200");
    return true;
}

bool SttClient::playResponseAudio(Speaker& speaker)
{
    WiFiClient client;
    client.setTimeout(HTTP_TIMEOUT_MS);
    Serial.println("[AUDIO] download start");

    if (!client.connect(STT_SERVER_HOST, STT_SERVER_PORT) ||
        client.printf("GET %s HTTP/1.1\r\nHost: %s:%u\r\nConnection: close\r\n\r\n",
                      RESPONSE_AUDIO_PATH,
                      STT_SERVER_HOST,
                      STT_SERVER_PORT) <= 0)
    {
        Serial.println("[AUDIO] download failed");
        client.stop();
        return false;
    }

    size_t wavSize = 0;
    if (readResponseHeaders(client, wavSize) != 200 || wavSize == 0 || wavSize > MAX_RESPONSE_AUDIO_BYTES)
    {
        Serial.println("[AUDIO] download failed");
        client.stop();
        return false;
    }

    uint8_t* const wavData = static_cast<uint8_t*>(heap_caps_malloc(wavSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (wavData == nullptr)
    {
        Serial.println("[AUDIO] download failed");
        client.stop();
        return false;
    }

    size_t receivedBytes = 0;
    uint32_t downloadChunks = 0;
    uint32_t shortReads = 0;
    while (receivedBytes < wavSize)
    {
        const size_t requestedBytes = min(RESPONSE_AUDIO_BLOCK_BYTES, wavSize - receivedBytes);
        const size_t chunkBytes = client.readBytes(reinterpret_cast<char*>(wavData + receivedBytes), requestedBytes);
        if (chunkBytes == 0)
            break;

        if (chunkBytes < requestedBytes)
            ++shortReads;
        ++downloadChunks;
        receivedBytes += chunkBytes;
    }
    client.stop();

    if (receivedBytes != wavSize)
    {
        heap_caps_free(wavData);
        Serial.println("[AUDIO] download failed");
        return false;
    }

    WavInfo wavInfo;
    if (!parseWav(wavData, wavSize, wavInfo) || wavInfo.sampleRate != RESPONSE_AUDIO_SAMPLE_RATE ||
        wavInfo.bitsPerSample != 16 || wavInfo.channels != 1 || wavInfo.pcmByteCount == 0 ||
        wavInfo.pcmByteCount % sizeof(int16_t) != 0)
    {
        heap_caps_free(wavData);
        Serial.println("[AUDIO] unsupported wav");
        return false;
    }

    Serial.printf("[AUDIO] wav rate=%lu bits=%u channels=%u\n",
                  static_cast<unsigned long>(wavInfo.sampleRate),
                  static_cast<unsigned>(wavInfo.bitsPerSample),
                  static_cast<unsigned>(wavInfo.channels));
    if (!speaker.beginPcmPlayback(wavInfo.sampleRate, wavInfo.bitsPerSample, wavInfo.channels))
    {
        heap_caps_free(wavData);
        Serial.println("[AUDIO] playback failed");
        return false;
    }

    const uint32_t expectedMs = static_cast<uint32_t>(wavInfo.pcmByteCount * 1000UL /
                                                      (wavInfo.sampleRate * wavInfo.channels * sizeof(int16_t)));
    const uint32_t playbackStartedAt = millis();
    size_t playedBytes = 0;
    uint32_t playbackChunks = 0;
    uint32_t writeFailures = 0;
    while (playedBytes < wavInfo.pcmByteCount)
    {
        const size_t chunkBytes = min(RESPONSE_AUDIO_BLOCK_BYTES, wavInfo.pcmByteCount - playedBytes);
        const int16_t* const samples = reinterpret_cast<const int16_t*>(wavInfo.pcmData + playedBytes);
        if (!speaker.playMonoPcm(samples, chunkBytes / sizeof(int16_t)))
        {
            ++writeFailures;
            speaker.stop();
            heap_caps_free(wavData);
            Serial.println("[AUDIO] playback failed");
            return false;
        }

        ++playbackChunks;
        playedBytes += chunkBytes;
    }

    speaker.stop();
    const uint32_t playbackMs = millis() - playbackStartedAt;
    heap_caps_free(wavData);
    Serial.printf("[AUDIO] expected_ms=%lu playback_ms=%lu chunks=%lu short_reads=%lu write_failures=%lu\n",
                  static_cast<unsigned long>(expectedMs),
                  static_cast<unsigned long>(playbackMs),
                  static_cast<unsigned long>(playbackChunks),
                  static_cast<unsigned long>(shortReads),
                  static_cast<unsigned long>(writeFailures));
    Serial.printf("[AUDIO] download_chunks=%lu\n", static_cast<unsigned long>(downloadChunks));
    Serial.println("[AUDIO] playback end");
    return true;
}

bool SttClient::uploadDataset(const int16_t* pcm16, size_t pcmByteCount, const char* label, uint32_t& index)
{
    WiFiClient client;
    client.setTimeout(HTTP_TIMEOUT_MS);
    if (!client.connect(STT_SERVER_HOST, STT_SERVER_PORT) ||
        !sendDatasetRequest(client, pcm16, pcmByteCount, label))
    {
        client.stop();
        return false;
    }

    const int statusCode = readStatusCode(client);
    const String body = readResponseBody(client);
    client.stop();
    return statusCode == 200 && extractIndex(body, index);
}

bool SttClient::uploadCalibration(const int16_t* pcm16,
                                  size_t pcmByteCount,
                                  const CalibrationMetadata& metadata,
                                  String& filename)
{
    WiFiClient client;
    client.setTimeout(HTTP_TIMEOUT_MS);
    if (!client.connect(STT_SERVER_HOST, STT_SERVER_PORT) ||
        !sendCalibrationRequest(client, pcm16, pcmByteCount, metadata))
    {
        Serial.println("[CAL] upload error=connection");
        client.stop();
        return false;
    }

    const int statusCode = readStatusCode(client);
    const String body = readResponseBody(client);
    client.stop();
    if (statusCode != 200 || !extractJsonString(body, "filename", filename))
    {
        Serial.printf("[CAL] upload error=http_%d\n", statusCode);
        return false;
    }

    return true;
}

bool SttClient::sendGetRequest(WiFiClient& client) const
{
    const int written = client.printf(
        "GET /health HTTP/1.1\r\n"
        "Host: %s:%u\r\n"
        "X-Rocky-Device: %s\r\n"
        "Connection: close\r\n\r\n",
        STT_SERVER_HOST,
        STT_SERVER_PORT,
        DEVICE_ID);
    return written > 0;
}

bool SttClient::sendMultipartRequest(WiFiClient& client,
                                     const int16_t* pcm16,
                                     size_t pcmByteCount) const
{
    const size_t contentLength = sizeof(MULTIPART_PREFIX) - 1 + WAV_HEADER_SIZE + pcmByteCount +
                                 sizeof(MULTIPART_SUFFIX) - 1;
    const int headerWritten = client.printf(
        "POST /transcribe HTTP/1.1\r\n"
        "Host: %s:%u\r\n"
        "X-Rocky-Device: %s\r\n"
        "Content-Type: multipart/form-data; boundary=%s\r\n"
        "Content-Length: %u\r\n"
        "Connection: close\r\n\r\n",
        STT_SERVER_HOST,
        STT_SERVER_PORT,
        DEVICE_ID,
        MULTIPART_BOUNDARY,
        static_cast<unsigned>(contentLength));
    if (headerWritten <= 0 ||
        !writeAll(client,
                  reinterpret_cast<const uint8_t*>(MULTIPART_PREFIX),
                  sizeof(MULTIPART_PREFIX) - 1))
    {
        return false;
    }

    uint8_t wavHeader[WAV_HEADER_SIZE]{};
    createWavHeader(wavHeader, pcmByteCount);
    if (!writeAll(client, wavHeader, sizeof(wavHeader)))
        return false;

    const uint8_t* const pcmBytes = reinterpret_cast<const uint8_t*>(pcm16);
    for (size_t offset = 0; offset < pcmByteCount; offset += PCM_WRITE_CHUNK_BYTES)
    {
        const size_t chunkSize = min(PCM_WRITE_CHUNK_BYTES, pcmByteCount - offset);
        if (!writeAll(client, pcmBytes + offset, chunkSize))
            return false;
    }

    return writeAll(client,
                    reinterpret_cast<const uint8_t*>(MULTIPART_SUFFIX),
                    sizeof(MULTIPART_SUFFIX) - 1);
}

bool SttClient::sendDatasetRequest(WiFiClient& client,
                                   const int16_t* pcm16,
                                   size_t pcmByteCount,
                                   const char* label) const
{
    const String prefix = String("------RockyBoundary\r\n") +
                          "Content-Disposition: form-data; name=\"label\"\r\n\r\n" + label +
                          "\r\n------RockyBoundary\r\n"
                          "Content-Disposition: form-data; name=\"file\"; filename=\"rocky.wav\"\r\n"
                          "Content-Type: audio/wav\r\n\r\n";
    const size_t contentLength = prefix.length() + WAV_HEADER_SIZE + pcmByteCount + sizeof(MULTIPART_SUFFIX) - 1;
    if (client.printf("POST /dataset HTTP/1.1\r\nHost: %s:%u\r\nContent-Type: multipart/form-data; boundary=%s\r\nContent-Length: %u\r\nConnection: close\r\n\r\n",
                      STT_SERVER_HOST, STT_SERVER_PORT, MULTIPART_BOUNDARY, static_cast<unsigned>(contentLength)) <= 0 ||
        !writeAll(client, reinterpret_cast<const uint8_t*>(prefix.c_str()), prefix.length()))
        return false;

    uint8_t wavHeader[WAV_HEADER_SIZE]{};
    createWavHeader(wavHeader, pcmByteCount);
    if (!writeAll(client, wavHeader, sizeof(wavHeader)))
        return false;

    const uint8_t* const pcmBytes = reinterpret_cast<const uint8_t*>(pcm16);
    for (size_t offset = 0; offset < pcmByteCount; offset += PCM_WRITE_CHUNK_BYTES)
    {
        const size_t chunkSize = min(PCM_WRITE_CHUNK_BYTES, pcmByteCount - offset);
        if (!writeAll(client, pcmBytes + offset, chunkSize))
            return false;
    }
    return writeAll(client, reinterpret_cast<const uint8_t*>(MULTIPART_SUFFIX), sizeof(MULTIPART_SUFFIX) - 1);
}

bool SttClient::sendCalibrationRequest(WiFiClient& client,
                                       const int16_t* pcm16,
                                       size_t pcmByteCount,
                                       const CalibrationMetadata& metadata) const
{
    String prefix;
    prefix.reserve(1800);
    appendMultipartField(prefix, "mode", metadata.mode);
    appendMultipartField(prefix, "distance_cm", String(metadata.distanceCm));
    appendMultipartField(prefix, "sample_id", String(metadata.sampleId));
    appendMultipartField(prefix, "voice_level", metadata.voiceLevel);
    appendMultipartField(prefix, "fan_state", metadata.fanState);
    appendMultipartField(prefix, "noise_rms", String(metadata.noiseRms));
    appendMultipartField(prefix, "noise_peak_rms", String(metadata.noisePeakRms));
    appendMultipartField(prefix, "raw24_rms", String(metadata.raw24Rms));
    appendMultipartField(prefix, "pcm16_rms", String(metadata.pcm16Rms));
    appendMultipartField(prefix, "peak_abs", String(metadata.peakAbsolute));
    appendMultipartField(prefix, "clipping", String(metadata.clippingCount));
    appendMultipartField(prefix, "vad_trigger_rms", String(metadata.vadTriggerRms));
    appendMultipartField(prefix, "vad_trigger_delay_ms", String(metadata.vadTriggerDelayMs));
    prefix += "------RockyBoundary\r\n"
              "Content-Disposition: form-data; name=\"file\"; filename=\"calibration.wav\"\r\n"
              "Content-Type: audio/wav\r\n\r\n";

    const size_t contentLength = prefix.length() + WAV_HEADER_SIZE + pcmByteCount + sizeof(MULTIPART_SUFFIX) - 1;
    if (client.printf("POST /calibration/audio HTTP/1.1\r\nHost: %s:%u\r\nContent-Type: multipart/form-data; boundary=%s\r\nContent-Length: %u\r\nConnection: close\r\n\r\n",
                      STT_SERVER_HOST, STT_SERVER_PORT, MULTIPART_BOUNDARY,
                      static_cast<unsigned>(contentLength)) <= 0 ||
        !writeAll(client, reinterpret_cast<const uint8_t*>(prefix.c_str()), prefix.length()))
    {
        return false;
    }

    uint8_t wavHeader[WAV_HEADER_SIZE]{};
    createWavHeader(wavHeader, pcmByteCount);
    if (!writeAll(client, wavHeader, sizeof(wavHeader)))
        return false;

    const uint8_t* const pcmBytes = reinterpret_cast<const uint8_t*>(pcm16);
    for (size_t offset = 0; offset < pcmByteCount; offset += PCM_WRITE_CHUNK_BYTES)
    {
        const size_t chunkSize = min(PCM_WRITE_CHUNK_BYTES, pcmByteCount - offset);
        if (!writeAll(client, pcmBytes + offset, chunkSize))
            return false;
    }

    return writeAll(client, reinterpret_cast<const uint8_t*>(MULTIPART_SUFFIX), sizeof(MULTIPART_SUFFIX) - 1);
}

bool SttClient::writeAll(WiFiClient& client, const uint8_t* data, size_t dataSize)
{
    size_t offset = 0;
    while (offset < dataSize)
    {
        const size_t written = client.write(data + offset, dataSize - offset);
        if (written == 0)
            return false;

        offset += written;
    }

    return true;
}

int SttClient::readStatusCode(WiFiClient& client)
{
    const uint32_t startedAt = millis();
    while (!client.available())
    {
        if (millis() - startedAt >= HTTP_TIMEOUT_MS)
            return 0;

        yield();
    }

    const String statusLine = client.readStringUntil('\n');
    if (!statusLine.startsWith("HTTP/"))
        return 0;

    while (millis() - startedAt < HTTP_TIMEOUT_MS)
    {
        if (!client.available())
        {
            yield();
            continue;
        }

        const String header = client.readStringUntil('\n');
        if (header == "\r" || header.isEmpty())
            return statusLine.substring(9, 12).toInt();
    }

    return 0;
}

int SttClient::readResponseHeaders(WiFiClient& client, size_t& contentLength)
{
    contentLength = 0;
    const uint32_t startedAt = millis();
    while (!client.available())
    {
        if (millis() - startedAt >= HTTP_TIMEOUT_MS)
            return 0;

        yield();
    }

    const String statusLine = client.readStringUntil('\n');
    if (!statusLine.startsWith("HTTP/"))
        return 0;

    while (millis() - startedAt < HTTP_TIMEOUT_MS)
    {
        if (!client.available())
        {
            yield();
            continue;
        }

        const String header = client.readStringUntil('\n');
        if (header == "\r" || header.isEmpty())
            return statusLine.substring(9, 12).toInt();

        const int separatorPosition = header.indexOf(':');
        if (separatorPosition < 0)
            continue;

        String name = header.substring(0, separatorPosition);
        name.toLowerCase();
        if (name == "content-length")
            contentLength = static_cast<size_t>(header.substring(separatorPosition + 1).toInt());
    }

    return 0;
}

String SttClient::readResponseBody(WiFiClient& client)
{
    String body;
    body.reserve(MAX_RESPONSE_BYTES);
    const uint32_t startedAt = millis();

    while (client.connected() || client.available())
    {
        while (client.available())
        {
            if (body.length() >= MAX_RESPONSE_BYTES)
                return body;

            body += static_cast<char>(client.read());
        }

        if (millis() - startedAt >= HTTP_TIMEOUT_MS)
            break;

        yield();
    }

    return body;
}

bool SttClient::parseWav(const uint8_t* wavData, size_t wavSize, WavInfo& wavInfo)
{
    wavInfo = {};
    if (wavData == nullptr || wavSize < 12 || memcmp(wavData, "RIFF", 4) != 0 ||
        memcmp(wavData + 8, "WAVE", 4) != 0)
        return false;

    bool formatFound = false;
    bool dataFound = false;
    size_t offset = 12;
    while (offset + 8 <= wavSize)
    {
        const uint8_t* const chunkId = wavData + offset;
        const size_t chunkSize = readLittleEndian32(wavData + offset + 4);
        const size_t chunkDataOffset = offset + 8;
        if (chunkSize > wavSize - chunkDataOffset)
            return false;

        if (memcmp(chunkId, "fmt ", 4) == 0)
        {
            if (chunkSize < 16 || readLittleEndian16(wavData + chunkDataOffset) != 1)
                return false;

            wavInfo.channels = readLittleEndian16(wavData + chunkDataOffset + 2);
            wavInfo.sampleRate = readLittleEndian32(wavData + chunkDataOffset + 4);
            wavInfo.bitsPerSample = readLittleEndian16(wavData + chunkDataOffset + 14);
            formatFound = true;
        }
        else if (memcmp(chunkId, "data", 4) == 0)
        {
            wavInfo.pcmData = wavData + chunkDataOffset;
            wavInfo.pcmByteCount = chunkSize;
            dataFound = true;
        }

        offset = chunkDataOffset + chunkSize;
        if (chunkSize % 2 != 0)
        {
            if (offset == wavSize)
                return false;
            ++offset;
        }
    }

    return formatFound && dataFound;
}

bool SttClient::extractText(const String& json, String& text)
{
    return extractJsonString(json, "text", text);
}

bool SttClient::extractBackendResponse(const String& json, BackendResponse& response)
{
    response = {};
    if (!extractText(json, response.text) || !extractJsonString(json, "interaction_state", response.interactionState))
        return false;

    extractJsonUnsigned(json, "command_window_ms", response.commandWindowMs);
    extractJsonBoolean(json, "audio_available", response.audioAvailable);

    const int actionsPosition = json.indexOf("\"actions\"");
    const int actionStart = json.indexOf('{', actionsPosition);
    if (actionsPosition < 0 || actionStart < 0)
    {
        response.valid = true;
        return true;
    }

    const int actionEnd = json.indexOf('}', actionStart);
    if (actionEnd < 0)
        return false;

    const String actionJson = json.substring(actionStart, actionEnd + 1);
    String type;
    if (!extractJsonString(actionJson, "type", type))
        return false;

    if (type == "expression")
    {
        response.action.type = BackendAction::Type::Expression;
        if (!extractJsonString(actionJson, "value", response.action.value))
            return false;
    }
    else if (type == "show_text")
    {
        response.action.type = BackendAction::Type::ShowText;
        if (!extractJsonString(actionJson, "line1", response.action.line1) ||
            !extractJsonString(actionJson, "line2", response.action.line2) ||
            !extractJsonUnsigned(actionJson, "duration_ms", response.action.durationMs))
            return false;
    }
    else
        response.action.value = type;

    response.valid = true;
    return true;
}

bool SttClient::extractIndex(const String& json, uint32_t& index)
{
    const int keyPosition = json.indexOf("\"index\"");
    const int separatorPosition = json.indexOf(':', keyPosition);
    if (keyPosition < 0 || separatorPosition < 0)
        return false;

    index = static_cast<uint32_t>(json.substring(separatorPosition + 1).toInt());
    return index > 0;
}

bool SttClient::isHealthyResponse(const String& json)
{
    String status;
    return extractJsonString(json, "status", status) && status == "ok";
}

bool SttClient::extractJsonString(const String& json, const char* key, String& value)
{
    const String quotedKey = String('"') + key + '"';
    const int keyPosition = json.indexOf(quotedKey);
    if (keyPosition < 0)
        return false;

    const int separatorPosition = json.indexOf(':', keyPosition + quotedKey.length());
    const int valueStart = json.indexOf('"', separatorPosition + 1);
    if (separatorPosition < 0 || valueStart < 0)
        return false;

    const int valueEnd = json.indexOf('"', valueStart + 1);
    if (valueEnd < 0)
        return false;

    value = json.substring(valueStart + 1, valueEnd);
    return true;
}

bool SttClient::extractJsonUnsigned(const String& json, const char* key, uint32_t& value)
{
    const String quotedKey = String('"') + key + '"';
    const int keyPosition = json.indexOf(quotedKey);
    const int separatorPosition = json.indexOf(':', keyPosition + quotedKey.length());
    if (keyPosition < 0 || separatorPosition < 0)
        return false;

    value = static_cast<uint32_t>(json.substring(separatorPosition + 1).toInt());
    return true;
}

bool SttClient::extractJsonBoolean(const String& json, const char* key, bool& value)
{
    const String quotedKey = String('"') + key + '"';
    const int keyPosition = json.indexOf(quotedKey);
    const int separatorPosition = json.indexOf(':', keyPosition + quotedKey.length());
    if (keyPosition < 0 || separatorPosition < 0)
        return false;

    const String valueText = json.substring(separatorPosition + 1);
    if (valueText.startsWith("true"))
    {
        value = true;
        return true;
    }
    if (valueText.startsWith("false"))
    {
        value = false;
        return true;
    }
    return false;
}

void SttClient::createWavHeader(uint8_t (&header)[44], size_t pcmByteCount)
{
    memcpy(header, "RIFF", 4);
    writeLittleEndian32(header + 4, static_cast<uint32_t>(pcmByteCount + 36));
    memcpy(header + 8, "WAVEfmt ", 8);
    writeLittleEndian32(header + 16, 16);
    writeLittleEndian16(header + 20, 1);
    writeLittleEndian16(header + 22, PCM_CHANNELS);
    writeLittleEndian32(header + 24, PCM_SAMPLE_RATE);
    writeLittleEndian32(header + 28, PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_BITS_PER_SAMPLE / 8);
    writeLittleEndian16(header + 32, PCM_CHANNELS * PCM_BITS_PER_SAMPLE / 8);
    writeLittleEndian16(header + 34, PCM_BITS_PER_SAMPLE);
    memcpy(header + 36, "data", 4);
    writeLittleEndian32(header + 40, static_cast<uint32_t>(pcmByteCount));
}
